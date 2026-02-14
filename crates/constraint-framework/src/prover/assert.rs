use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{
    bit_reverse_index, circle_domain_index_to_coset_index, coset_index_to_circle_domain_index,
};
use stwo::core::Fraction;
use stwo::parallel_iter;
use stwo::prover::backend::{Backend, Column};
use stwo::prover::poly::circle::CircleCoefficients;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX};

/// Evaluates expressions at a trace domain row, and asserts constraints. Mainly used for testing.
pub struct AssertEvaluator<'a> {
    pub trace: &'a TreeVec<Vec<&'a Vec<BaseField>>>,
    pub col_index: TreeVec<usize>,
    pub row: usize,
    pub constraint_counter: usize,
    pub logup: LogupAtRow<Self>,
}
impl<'a> AssertEvaluator<'a> {
    pub fn new(
        trace: &'a TreeVec<Vec<&Vec<BaseField>>>,
        row: usize,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace,
            col_index: TreeVec::new(vec![0; trace.len()]),
            row,
            constraint_counter: 0,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}
impl EvalAtRow for AssertEvaluator<'_> {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                let col = &self.trace[interaction][col_index];
                return col[self.row];
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            let log_size = self.logup.log_size;
            let domain_size = 1 << log_size;

            let coset_index =
                circle_domain_index_to_coset_index(bit_reverse_index(self.row, log_size), log_size);
            let next_coset_index = (coset_index as isize + off).rem_euclid(domain_size);
            let next_index = bit_reverse_index(
                coset_index_to_circle_domain_index(next_coset_index as usize, log_size),
                log_size,
            );
            self.trace[interaction][col_index].at(next_index)
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF> + From<G>,
    {
        // Cast to SecureField.
        // The constraint should be zero at the given row, since we are evaluating on the trace
        // domain.

        let constraint_val = Self::EF::from(constraint);

        // // DEBUG: Uncomment to print constraint details for row 0
        // if self.row == 0 && self.constraint_counter <= 20 {
        //         "=== DEBUG CPU: Row {}, Constraint #{} === constraint_val: {:?}",
        //         self.row,
        //         self.constraint_counter,
        //         constraint_val
        //     );
        // }

        assert_eq!(
            constraint_val,
            SecureField::zero(),
            "row: #{}, constraint #{}",
            self.row,
            self.constraint_counter
        );

        self.constraint_counter += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_m31_array(values)
    }

    fn add_to_relation<R: crate::Relation<Self::F, Self::EF>>(
        &mut self,
        entry: crate::RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let frac = Fraction::new(
            entry.multiplicity.clone(),
            entry.relation.combine(entry.values),
        );

        self.write_logup_frac(frac);
    }

    crate::logup_proxy!();
}

pub fn assert_constraints_on_polys<B: Backend>(
    trace_polys: &TreeVec<Vec<CircleCoefficients<B>>>,
    trace_domain: CanonicCoset,
    assert_func: impl Fn(AssertEvaluator<'_>) + Sync,
    claimed_sum: SecureField,
) {
    let traces = trace_polys.as_ref().map(|tree| {
        tree.iter()
            .map(|poly| poly.evaluate(trace_domain.circle_domain()).values.to_cpu())
            .collect_vec()
    });
    let traces = &traces.as_ref();
    let traces = traces.into();
    assert_constraints_on_trace(&traces, trace_domain.log_size(), assert_func, claimed_sum);
}

pub fn assert_constraints_on_trace(
    evals: &TreeVec<Vec<&Vec<M31>>>,
    log_size: u32,
    assert_func: impl Fn(AssertEvaluator<'_>) + Sync,
    claimed_sum: SecureField,
) {
    let n_rows = 1 << log_size;

    let iter = parallel_iter!(0..n_rows);
    iter.for_each(|row| {
        let eval = AssertEvaluator::new(evals, row, log_size, claimed_sum);
        assert_func(eval);
    });
}

/// Asserts CUDA constraint quotients match CPU-computed reference.
///
/// This function calls the CUDA kernel `evaluate_constraint_quotients_on_domain` to compute
/// constraint quotients, then verifies them against CPU-computed quotients on sample points.
/// This provides strong evidence that the CUDA kernel is computing constraints correctly.
///
/// # Arguments
/// * `trace_polys` - CUDA trace polynomials (one tree per commitment phase)
/// * `trace_domain` - The canonical coset trace domain
/// * `eval_log_size` - Log size of the evaluation domain (typically trace_log_size + LOG_EXPAND)
/// * `evaluator` - The evaluator implementing FrameworkEval (e.g., PoseidonEval)
/// * `n_constraints` - Number of constraints
/// * `claimed_sum` - The claimed sum for the logup protocol
/// * `logup_counts` - Number of logup entries per row
///
/// # Panics
/// Panics if CUDA quotients don't match CPU reference quotients on sampled points.
pub fn assert_constraints_on_polys_cuda<E: crate::FrameworkEval + Sync>(
    trace_polys: &TreeVec<Vec<CircleCoefficients<stwo::prover::backend::cuda::CudaBackend>>>,
    trace_domain: CanonicCoset,
    eval_log_size: u32,
    evaluator: &E,
    n_constraints: usize,
    claimed_sum: SecureField,
    logup_counts: u32,
) {
    use num_traits::One;
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::backend::{Col, Column};
    use stwo::core::constraints::coset_vanishing;
    use stwo::core::utils::bit_reverse;
    use stwo::stwo_cuda::{BaseFieldVec, SecureFieldVec, bindings};
    use stwo::stwo_cuda::bindings::CudaSecureField;
    #[allow(unused_imports)]
    use stwo::prover::poly::circle::PolyOps;

    let trace_log_size = trace_domain.log_size();

    // Create evaluation domain from the provided log size
    let eval_domain = CanonicCoset::new(eval_log_size).circle_domain();

    // Evaluate trace polynomials on the evaluation domain
    let trace_evals: TreeVec<Vec<_>> = trace_polys.as_ref().map(|tree| {
        tree.iter()
            .map(|poly| poly.evaluate(eval_domain))
            .collect_vec()
    });

    // Compute denominator inverses only for the expansion factor
    let log_expand = eval_log_size - trace_log_size;
    let mut denom_inv = (0..1 << log_expand)
        .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
        .collect_vec();
    bit_reverse(&mut denom_inv);
    let denom_inv_cpu = denom_inv.clone(); // Keep a CPU copy for verification
    let gpu_denom_inv = BaseFieldVec::from_vec(denom_inv);

    // Allocate quotient columns on GPU (one SecureField = 4 BaseField components)
    let quotient_size = 1 << eval_log_size;
    let quotients_0 = Col::<CudaBackend, BaseField>::zeros(quotient_size);
    let quotients_1 = Col::<CudaBackend, BaseField>::zeros(quotient_size);
    let quotients_2 = Col::<CudaBackend, BaseField>::zeros(quotient_size);
    let quotients_3 = Col::<CudaBackend, BaseField>::zeros(quotient_size);

    // Random coefficients (set to 1 for assertion - we check each constraint)
    let random_coeffs = vec![SecureField::one(); n_constraints];
    let random_coeff_powers = SecureFieldVec::from_vec(random_coeffs);

    // Prepare trace evaluation pointers
    let trace0_ptrs: Vec<_> = trace_evals.get(0).map_or(vec![], |t| {
        t.iter().map(|eval| eval.values.device_ptr).collect()
    });
    let trace1_ptrs: Vec<_> = trace_evals.get(1).map_or(vec![], |t| {
        t.iter().map(|eval| eval.values.device_ptr).collect()
    });
    let trace2_ptrs: Vec<_> = trace_evals.get(2).map_or(vec![], |t| {
        t.iter().map(|eval| eval.values.device_ptr).collect()
    });

    // Call CUDA kernel to evaluate constraints
    unsafe {
        let eval_ptr = evaluator as *const E as *mut std::os::raw::c_void;
        bindings::evaluate_constraint_quotients_on_domain(
            quotients_0.device_ptr,
            quotients_1.device_ptr,
            quotients_2.device_ptr,
            quotients_3.device_ptr,
            trace0_ptrs.as_ptr(),
            trace0_ptrs.len() as u32,
            trace1_ptrs.as_ptr(),
            trace1_ptrs.len() as u32,
            trace2_ptrs.as_ptr(),
            trace2_ptrs.len() as u32,
            random_coeff_powers.device_ptr,
            gpu_denom_inv.device_ptr,
            trace_log_size,
            eval_log_size,
            n_constraints as u32,
            logup_counts,
            eval_ptr,
            CudaSecureField::from(claimed_sum / BaseField::from_u32_unchecked(1 << trace_log_size)),
            false,
            true,  // use_assert_evaluator = true for tests
        );
    }

    // Verify CUDA quotient results by computing reference quotients on CPU
    // Transfer CUDA quotients to CPU
    let cuda_q0 = quotients_0.to_cpu();
    let cuda_q1 = quotients_1.to_cpu();
    let cuda_q2 = quotients_2.to_cpu();
    let cuda_q3 = quotients_3.to_cpu();

    // Compute CPU reference quotients for verification
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::prover::poly::circle::CircleEvaluation;
    use stwo::prover::poly::BitReversedOrder;
    use super::CpuDomainEvaluator;

    // Convert CUDA trace polynomials to CPU
    let cpu_trace_polys: TreeVec<Vec<CircleCoefficients<SimdBackend>>> = trace_polys.as_ref().map(|tree| {
        tree.iter()
            .map(|cuda_poly| {
                // Transfer coefficients from GPU to CPU
                let cpu_coeffs = BaseColumn::from_cpu(&cuda_poly.coeffs.to_cpu());
                CircleCoefficients::new(cpu_coeffs)
            })
            .collect_vec()
    });

    // Evaluate CPU trace on the extended evaluation domain
    let cpu_trace_evals: TreeVec<Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>> =
        cpu_trace_polys.as_ref().map(|tree| {
            tree.iter()
                .map(|poly| poly.evaluate(eval_domain))
                .collect_vec()
        });

    // Prepare CPU trace references for evaluation
    let cpu_trace_refs = cpu_trace_evals.as_cols_ref().map_cols(|c| c.to_cpu());
    let trace_cols = cpu_trace_refs.as_cols_ref();

    // Random coefficients (same as CUDA: all ones)
    let random_coeff_powers = vec![SecureField::one(); n_constraints];

    // Sample points to verify (checking all points would be expensive)
    let eval_size = 1 << eval_log_size;
    let sample_indices = if eval_size <= 16 {
        // For small domains, check all points
        (0..eval_size).collect::<Vec<_>>()
    } else {
        // For larger domains, sample first, middle, last and a few random points
        vec![0, 1, eval_size / 4, eval_size / 2, 3 * eval_size / 4, eval_size - 2, eval_size - 1]
    };

    let mut mismatches = 0;
    for &row in &sample_indices {
        // Create CPU evaluator for this row
        let eval = CpuDomainEvaluator::new(
            &trace_cols,
            row,
            &random_coeff_powers,
            trace_log_size,
            eval_log_size,
            evaluator.log_size(),
            claimed_sum,
        );

        // Evaluate constraints using the provided evaluator
        let cpu_eval_result = evaluator.evaluate(eval);
        let cpu_constraint_eval = cpu_eval_result.row_res;

        // Divide by vanishing polynomial to get quotient
        let denom_inv_idx = row >> trace_log_size;
        let denom_inv_value = denom_inv_cpu[denom_inv_idx];
        let cpu_quotient = cpu_constraint_eval * denom_inv_value;

        // Get CUDA quotient
        let cuda_quotient = SecureField::from_m31_array([
            cuda_q0[row],
            cuda_q1[row],
            cuda_q2[row],
            cuda_q3[row]
        ]);

        // Compare
        if cuda_quotient != cpu_quotient {
            mismatches += 1;
            if mismatches <= 3 {
                println!("Mismatch at row {}: CUDA={:?}, CPU={:?}",
                    row, cuda_quotient, cpu_quotient);
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "CUDA quotients differ from CPU reference at {} out of {} sampled points",
        mismatches, sample_indices.len()
    );

    println!("✓ CUDA constraint quotients verified against CPU reference");
    println!("  Verified {} sample points, all quotients match exactly", sample_indices.len());
}
