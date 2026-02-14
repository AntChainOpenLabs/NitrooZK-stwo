use std::ffi::c_void;

use crate::core::fields::{m31::BaseField, qm31::SecureField};
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch};
use crate::core::poly::circle::CanonicCoset;
use crate::prover::backend::Column;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

use crate::stwo_cuda::bindings;
use crate::stwo_cuda::bindings::{CirclePointSecureField, CudaSecureField};
use crate::prover::backend::cuda::CudaBackend;

impl QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].len();

        // Precompute line coefficients on CPU (small data, sequential dependency)
        let quotient_consts = quotient_constants(sample_batches);

        // Upload column pointers to GPU (columns are already on GPU)
        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.values.device_ptr).collect();
        let col_ptrs_device = unsafe {
            bindings::copy_device_pointer_vec_from_host_to_device(
                col_ptrs.as_ptr(),
                col_ptrs.len(),
            )
        };

        for (batch, coeffs) in sample_batches.iter().zip(quotient_consts.line_coeffs.iter()) {
            // Extract b, c coefficients and column indices
            let line_coeffs_b: Vec<CudaSecureField> = coeffs
                .iter()
                .map(|(_, b, _)| CudaSecureField::from(*b))
                .collect();
            let line_coeffs_c: Vec<CudaSecureField> = coeffs
                .iter()
                .map(|(_, _, c)| CudaSecureField::from(*c))
                .collect();
            let column_indices: Vec<u32> = batch
                .cols_vals_randpows
                .iter()
                .map(|nd| nd.column_index as u32)
                .collect();
            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, _, _)| a).sum();

            // Allocate GPU result
            let result = SecureColumnByCoords::<CudaBackend>::zeros(size);

            unsafe {
                bindings::accumulate_numerators_batch(
                    size as u32,
                    col_ptrs_device as *const *const u32,
                    line_coeffs_b.as_ptr(),
                    line_coeffs_c.as_ptr(),
                    column_indices.as_ptr(),
                    coeffs.len() as u32,
                    result.columns[0].device_ptr,
                    result.columns[1].device_ptr,
                    result.columns[2].device_ptr,
                    result.columns[3].device_ptr,
                );
            }

            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc: result,
                first_linear_term_acc,
            });
        }

        unsafe {
            bindings::cuda_free_memory(col_ptrs_device as *const c_void);
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        _lifting_log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let max_log_size = accs
            .iter()
            .map(|x| x.partial_numerators_acc.len())
            .max()
            .unwrap()
            .ilog2();
        let max_size = 1u32 << max_log_size;
        let domain = CanonicCoset::new(max_log_size).circle_domain();

        // Compute half_coset parameters
        let half_coset = domain.half_coset;
        let half_coset_initial_index = half_coset.initial_index.0 as u32;
        let half_coset_step_size = half_coset.step_size.0 as u32;

        let num_acc = accs.len();

        // Flatten partial column pointers: [acc0_col0, acc0_col1, acc0_col2, acc0_col3, acc1_col0, ...]
        let mut acc_partial_columns: Vec<*const u32> = Vec::with_capacity(num_acc * 4);
        let mut acc_log_sizes: Vec<i32> = Vec::with_capacity(num_acc);
        let mut first_linear_term_accs: Vec<CudaSecureField> = Vec::with_capacity(num_acc);
        let mut sample_points: Vec<CirclePointSecureField> = Vec::with_capacity(num_acc);

        for acc in &accs {
            for col in &acc.partial_numerators_acc.columns {
                acc_partial_columns.push(col.device_ptr);
            }
            acc_log_sizes.push(acc.partial_numerators_acc.len().ilog2() as i32);
            first_linear_term_accs.push(CudaSecureField::from(acc.first_linear_term_acc));
            sample_points.push(CirclePointSecureField::from(acc.sample_point));
        }

        // Allocate output
        let result = SecureColumnByCoords::<CudaBackend>::zeros(max_size as usize);

        unsafe {
            bindings::compute_quotients_and_combine(
                max_size,
                max_log_size,
                half_coset_initial_index,
                half_coset_step_size,
                num_acc as u32,
                acc_partial_columns.as_ptr() as *const *const u32,
                acc_log_sizes.as_ptr(),
                first_linear_term_accs.as_ptr(),
                sample_points.as_ptr(),
                result.columns[0].device_ptr,
                result.columns[1].device_ptr,
                result.columns[2].device_ptr,
                result.columns[3].device_ptr,
            );
        }

        SecureEvaluation::new(domain, result)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::pcs::quotients::{ColumnSampleBatch, NumeratorData};
    use crate::prover::QuotientOps;
    use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::poly::circle::CircleEvaluation;
    use crate::prover::poly::BitReversedOrder;

    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::prover::backend::cuda::CudaBackend;
    use num_traits::Zero;

    #[test]
    fn test_accumulate_numerators_compared_with_cpu() {
        const LOG_SIZE: u32 = 5;
        let small_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let e0: BaseColumn = (0..small_domain.size()).map(BaseField::from).collect();
        let e1: BaseColumn = (0..small_domain.size())
            .map(|i| BaseField::from(2 * i))
            .collect();
        let polys = vec![
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                small_domain,
                BaseFieldVec::from_vec(e0.to_cpu()),
            )
            .interpolate(),
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                small_domain,
                BaseFieldVec::from_vec(e1.to_cpu()),
            )
            .interpolate(),
        ];
        let columns = vec![polys[0].evaluate(small_domain), polys[1].evaluate(small_domain)];
        let a = polys[0].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let b = polys[1].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
        let random_coeff = QM31::from_m31(M31::from(1), M31::from(2), M31::from(3), M31::from(4));
        let samples = vec![ColumnSampleBatch {
            point: SECURE_FIELD_CIRCLE_GEN,
            cols_vals_randpows: vec![
                NumeratorData {
                    column_index: 0,
                    sample_value: a,
                    random_coeff,
                },
                NumeratorData {
                    column_index: 1,
                    sample_value: b,
                    random_coeff,
                },
            ],
        }];

        let cpu_columns: Vec<_> = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect();

        let mut cpu_acc = vec![];
        CpuBackend::accumulate_numerators(
            &cpu_columns.iter().collect_vec(),
            &samples,
            &mut cpu_acc,
        );

        let mut gpu_acc = vec![];
        CudaBackend::accumulate_numerators(
            &columns.iter().collect_vec(),
            &samples,
            &mut gpu_acc,
        );

        assert_eq!(cpu_acc.len(), gpu_acc.len());
        for (c, g) in cpu_acc.iter().zip(gpu_acc.iter()) {
            assert_eq!(c.sample_point, g.sample_point);
            assert_eq!(c.first_linear_term_acc, g.first_linear_term_acc);
            let cpu_vals = c.partial_numerators_acc.to_vec();
            let gpu_vals = g.partial_numerators_acc.to_vec();
            assert_eq!(cpu_vals, gpu_vals);
        }
    }
}
