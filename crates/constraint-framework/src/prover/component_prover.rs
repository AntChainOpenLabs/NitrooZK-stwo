use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, Trace};
use stwo::stwo_cuda::{BaseFieldVec, SecureFieldVec, bindings};
use stwo::stwo_cuda::bindings::CudaSecureField;
use tracing::{span, Level};

use super::{CpuDomainEvaluator, SimdDomainEvaluator};
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

const CHUNK_SIZE: usize = 1;

impl<E: FrameworkEval + Sync> ComponentProver<SimdBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.eval.log_size());

        let mut component_polys = trace.polys.sub_tree(&self.trace_locations);
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        let mut component_evals = trace.evals.sub_tree(&self.trace_locations);
        component_evals[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.evals[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        // Extend trace if necessary.
        // TODO: Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain. (For larger blowup factors).
        let need_to_extend = component_evals
            .iter()
            .flatten()
            .any(|c| c.domain != eval_domain);
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Constraint Extension").entered();
            let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles)))
        } else {
            component_evals.clone().map_cols(|c| Cow::Borrowed(*c))
        };

        // Denom inverses.
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Accumulator.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval"
        )
        .entered();

        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
            // Fall back to CPU if the trace is too small.
            let mut col = accum.col.to_cpu();

            for row in 0..(1 << eval_domain.log_size()) {
                let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
                let trace_cols = trace_cols.as_cols_ref();

                // Evaluate constrains at row.
                let eval = CpuDomainEvaluator::new(
                    &trace_cols,
                    row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self.eval.log_size(),
                    self.claimed_sum,
                );
                let row_res = self.eval.evaluate(eval).row_res;

                // Finalize row.
                let denom_inv = denom_inv[row >> trace_domain.log_size()];
                col.set(row, col.at(row) + row_res * denom_inv)
            }
            let col = SecureColumnByCoords::from_cpu(col);
            *accum.col = col;
            return;
        }

        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(accum.col) };

        let range = 0..(1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS));

        #[cfg(not(feature = "parallel"))]
        let iter = range.step_by(CHUNK_SIZE).zip(col.chunks_mut(CHUNK_SIZE));

        #[cfg(feature = "parallel")]
        let iter = range
            .into_par_iter()
            .step_by(CHUNK_SIZE)
            .zip(col.chunks_mut(CHUNK_SIZE));

        // Define any `self` values outside the loop to prevent the compiler thinking there is a
        // `Sync` requirement on `Self`.
        let self_eval = &self.eval;
        let self_claimed_sum = self.claimed_sum;

        iter.for_each(|(chunk_idx, mut chunk)| {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

            for idx_in_chunk in 0..CHUNK_SIZE {
                let vec_row = chunk_idx * CHUNK_SIZE + idx_in_chunk;
                // Evaluate constrains at row.
                let eval = SimdDomainEvaluator::new(
                    &trace_cols,
                    vec_row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self_eval.log_size(),
                    self_claimed_sum,
                );
                let row_res = self_eval.evaluate(eval).row_res;

                // Finalize row.
                unsafe {
                    let denom_inv = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(
                        idx_in_chunk,
                        chunk.packed_at(idx_in_chunk) + row_res * denom_inv,
                    )
                }
            }
        });
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<CudaBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }
        let _span = span!(Level::INFO, "evaluate_constraint_quotients_on_domain").entered();

        let span = span!(Level::INFO, "prepare").entered();

        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let eval_log_size = self.eval.log_size();
        let trace_domain = CanonicCoset::new(eval_log_size);

        let mut component_polys = trace.polys.sub_tree(&self.trace_locations);
        span.exit();

        let span = span!(Level::INFO, "component_evals generate").entered();
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();
        span.exit();

        let span = span!(Level::INFO, "component_evals generate").entered();
        let mut component_evals = trace.evals.sub_tree(&self.trace_locations);
        component_evals[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.evals[PREPROCESSED_TRACE_IDX][*idx])
            .collect();
        span.exit();

        let span = span!(Level::INFO, "Check need_to_extend").entered();
        let need_to_extend = component_evals
            .iter()
            .flatten()
            .any(|c| c.domain != eval_domain);
        span.exit();

        let span = span!(Level::INFO, "Check & Extension").entered();
        #[cfg(not(feature = "parallel"))]
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = CudaBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| {
                    Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles))
                })
        } else {
            component_evals.clone().map_cols(|c| Cow::Borrowed(*c))
        };

        #[cfg(feature = "parallel")]
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = CudaBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .map_cols_par(|col| {
                    Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles))
                })
        } else {
            component_evals.clone().map_cols(|c| Cow::Borrowed(*c))
        };
        span.exit();

        // Denom inverses.
        let span = span!(Level::INFO, "Denom inverses generate").entered();
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        let gpu_denom_inv = BaseFieldVec::from_vec(denom_inv);
        span.exit();

        let span = span!(Level::INFO, "Accumulator").entered();

        // Accumulator.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();
        let random_coeff_powers = SecureFieldVec::from_vec(accum.random_coeff_powers);

        let trace0_evaluations_vec = trace[0]
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let trace1_evaluations_vec = trace[1]
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let mut trace2_evaluations_vec = vec![];
        if trace.len() != 2 {
            trace2_evaluations_vec = trace[2]
                .iter()
                .map(|column_evaluations| column_evaluations.device_ptr)
                .collect_vec();
        }
        span.exit();

        let span = span!(Level::INFO, "GPU evaluate_constraint_quotients_on_domain").entered();
        unsafe {
            let eval_ptr = &self.eval as *const _ as *mut std::os::raw::c_void;
            let logup_counts = self.logup_counts().values().sum::<usize>() as u32 / (1 << eval_log_size);

            bindings::evaluate_constraint_quotients_on_domain(
                accum.col.columns[0].device_ptr,
                accum.col.columns[1].device_ptr,
                accum.col.columns[2].device_ptr,
                accum.col.columns[3].device_ptr,
                trace0_evaluations_vec.as_ptr(),
                trace0_evaluations_vec.len() as u32,
                trace1_evaluations_vec.as_ptr(),
                trace1_evaluations_vec.len() as u32,
                trace2_evaluations_vec.as_ptr(),
                trace2_evaluations_vec.len() as u32,
                random_coeff_powers.device_ptr,
                gpu_denom_inv.device_ptr,
                trace_domain.log_size() as u32,
                eval_domain.log_size() as u32,
                self.n_constraints() as u32,
                logup_counts,
                eval_ptr,
                CudaSecureField::from(self.claimed_sum / BaseField::from_u32_unchecked(1 << eval_log_size)),
                true,   // should_accumulate: true for proving
                false,  // use_assert_evaluator: false for proving
            );
        };
        span.exit();
        return;
    }
}
