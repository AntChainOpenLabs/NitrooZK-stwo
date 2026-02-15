use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, Trace};
use stwo::stwo_cuda::{BaseFieldVec, SecureFieldVec, bindings};
use stwo::stwo_cuda::bindings::CudaSecureField;
use tracing::{span, Level};

use super::{CpuDomainEvaluator, SimdDomainEvaluator};
use crate::{fnv1a_eval_id_gen, FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

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

        if self.is_disabled() {
            evaluation_accumulator.skip_coeffs(self.n_constraints());
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

        // Extend trace if necessary.
        // TODO: Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain. (For larger blowup factors).
        let need_to_extend = component_polys
            .iter()
            .flatten()
            .any(|c| c.evals.domain.log_size() != eval_domain.log_size());
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Constraint Extension").entered();
            let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
            #[cfg(not(feature = "parallel"))]
            {
                component_polys.as_cols_ref().map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
            #[cfg(feature = "parallel")]
            {
                component_polys.as_cols_ref().par_map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
        } else {
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        };

        // Denom inverses.
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Note that `accum` is a mutable reference to a column in `evaluation_accumulator`.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval"
        )
        .entered();

        // Fall back to CPU if the trace is too small.
        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
            let trace_cols = trace_cols.as_cols_ref();
            *accum.col = SecureColumnByCoords::from_cpu(accumulate_pointwise_cpu(
                self,
                trace_cols,
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv,
                &accum.random_coeff_powers,
                &accum.col.to_cpu(),
            ));
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
                    let row_denom_inv = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(
                        idx_in_chunk,
                        chunk.packed_at(idx_in_chunk) + row_res * row_denom_inv,
                    )
                }
            }
        });
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<CpuBackend> for FrameworkComponent<E> {
    /// Almost all this implementation is equal to the one above for `SimdBackend`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
    ) {
        let n_constraints = self.n_constraints();
        if n_constraints == 0 {
            return;
        }

        if self.is_disabled() {
            evaluation_accumulator.skip_coeffs(n_constraints);
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

        // Extend trace if necessary.
        // TODO: Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain. (For larger blowup factors).
        let need_to_extend = component_polys
            .iter()
            .flatten()
            .any(|c| c.evals.domain.log_size() != eval_domain.log_size());
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Constraint Extension").entered();
            let twiddles = CpuBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles)))
        } else {
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
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
        let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

        *accum.col = accumulate_pointwise_cpu(
            self,
            trace_cols,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            accum.col,
        );
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

        if self.is_disabled() {
            evaluation_accumulator.skip_coeffs(self.n_constraints());
            return;
        }
        let _span = span!(Level::INFO, "evaluate_constraint_quotients_on_domain").entered();

        let span = span!(Level::INFO, "prepare").entered();

        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let eval_log_size = self.eval.log_size();
        let trace_domain = CanonicCoset::new(eval_log_size);

        let mut component_polys = trace.polys.sub_tree(&self.trace_locations);
        span.exit();

        let span = span!(Level::INFO, "component_polys generate").entered();
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();
        span.exit();

        let span = span!(Level::INFO, "Check need_to_extend").entered();
        let need_to_extend = component_polys
            .iter()
            .flatten()
            .any(|c| c.evals.domain.log_size() != eval_domain.log_size());
        span.exit();

        let span = span!(Level::INFO, "Check & Extension").entered();
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = CudaBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles)))
        } else {
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        };
        span.exit();

        // Denom inverses.
        let span = span!(Level::INFO, "Denom inverses generate").entered();
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);
        span.exit();

        let span = span!(Level::INFO, "Accumulator").entered();

        // Accumulator.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        // CPU fallback mode: evaluate constraints on CPU instead of CUDA.
        // CUDA_CPU_FALLBACK=1 → all components use CPU
        // CUDA_CPU_FALLBACK=component_name → only that component uses CUDA, rest use CPU
        // CUDA_ONLY=component_name → only that component uses CUDA, rest use CPU
        let use_cpu = match std::env::var("CUDA_CPU_FALLBACK").ok() {
            Some(val) if val == "1" || val.is_empty() => true,
            Some(component_name) => {
                // Use CPU for all EXCEPT the named component
                self.eval.cuda_eval_name() != component_name.as_str()
            }
            None => match std::env::var("CUDA_ONLY").ok() {
                Some(only_list) => {
                    // Use CUDA only for the listed components (comma-separated), CPU for rest
                    !only_list.split(',').any(|name| name.trim() == self.eval.cuda_eval_name())
                }
                None => match std::env::var("CUDA_EXCEPT").ok() {
                    Some(except_list) => {
                        // Use CPU for the listed components (comma-separated), CUDA for rest
                        except_list.split(',').any(|name| name.trim() == self.eval.cuda_eval_name())
                    }
                    None => false,
                },
            },
        };
        if use_cpu {
            let _span = span!(Level::INFO, "CPU fallback evaluation").entered();
            tracing::info!(
                component = self.eval.cuda_eval_name(),
                "Using CPU fallback for constraint evaluation"
            );

            // Download GPU trace columns to CPU.
            let cpu_trace = trace.as_cols_ref().map_cols(|c| {
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            });
            let cpu_trace_refs = cpu_trace.as_cols_ref();

            // Download current accumulator state from GPU.
            let accum_cpu = accum.col.to_cpu();

            // Evaluate on CPU.
            let result = accumulate_pointwise_cpu(
                self,
                cpu_trace_refs,
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv,
                &accum.random_coeff_powers,
                &accum_cpu,
            );

            // Upload result back to GPU.
            for (i, cpu_col) in result.columns.into_iter().enumerate() {
                accum.col.columns[i] = BaseFieldVec::from_vec(cpu_col);
            }
            span.exit();
            return;
        }

        // Clone data needed for potential CPU auto-fallback (if CUDA kernel is missing).
        let denom_inv_backup = denom_inv.clone();
        let random_coeff_powers_backup = accum.random_coeff_powers.clone();
        let gpu_denom_inv = BaseFieldVec::from_vec(denom_inv);
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
        let cuda_handled = unsafe {
            // Prepend eval_id (FNV1a hash of component name) before the Eval struct data.
            // The CUDA dispatcher reads eval_id from offset 0 to select the correct kernel.
            let eval_id = fnv1a_eval_id_gen(self.eval.cuda_eval_name());
            let eval_bytes = std::slice::from_raw_parts(
                &self.eval as *const _ as *const u8,
                std::mem::size_of::<E>(),
            );
            let mut cuda_eval_buffer = Vec::with_capacity(4 + eval_bytes.len());
            cuda_eval_buffer.extend_from_slice(&eval_id.to_ne_bytes());
            cuda_eval_buffer.extend_from_slice(eval_bytes);
            let eval_ptr = cuda_eval_buffer.as_ptr() as *mut std::os::raw::c_void;
            let logup_counts = self.logup_counts().values().sum::<usize>() as u32 / (1 << eval_log_size);

            tracing::info!(
                component = self.eval.cuda_eval_name(),
                eval_id = format!("{:#x}", eval_id),
                n_constraints = self.n_constraints(),
                logup_counts = logup_counts,
                trace0_len = trace0_evaluations_vec.len(),
                trace1_len = trace1_evaluations_vec.len(),
                trace2_len = trace2_evaluations_vec.len(),
                domain_log_size = trace_domain.log_size(),
                eval_domain_log_size = eval_domain.log_size(),
                "CUDA dispatch"
            );

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
            )
        };
        span.exit();

        if !cuda_handled {
            // CUDA dispatch doesn't have a kernel for this component yet.
            // Fall back to CPU evaluation automatically.
            let _span = span!(Level::INFO, "CPU auto-fallback (no CUDA kernel)").entered();
            tracing::warn!(
                component = self.eval.cuda_eval_name(),
                "No CUDA kernel available, using CPU fallback"
            );

            // Download GPU trace columns to CPU.
            let cpu_trace = trace.as_cols_ref().map_cols(|c| {
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            });
            let cpu_trace_refs = cpu_trace.as_cols_ref();

            // Download current accumulator state from GPU.
            let accum_cpu = accum.col.to_cpu();

            // Evaluate on CPU using the backed-up data.
            let result = accumulate_pointwise_cpu(
                self,
                cpu_trace_refs,
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv_backup,
                &random_coeff_powers_backup,
                &accum_cpu,
            );

            // Upload result back to GPU.
            for (i, cpu_col) in result.columns.into_iter().enumerate() {
                accum.col.columns[i] = BaseFieldVec::from_vec(cpu_col);
            }
        }
        return;
    }
}

fn accumulate_pointwise_cpu<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    trace_cols: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    eval_log_size: u32,
    trace_log_size: u32,
    denom_inv: Vec<BaseField>,
    random_coeff_powers: &[SecureField],
    accum: &SecureColumnByCoords<CpuBackend>,
) -> SecureColumnByCoords<CpuBackend> {
    let mut res = SecureColumnByCoords::zeros(1 << eval_log_size);
    for row in 0..(1 << eval_log_size) {
        // Evaluate constrains at row.
        let eval = CpuDomainEvaluator::new(
            &trace_cols,
            row,
            random_coeff_powers,
            trace_log_size,
            eval_log_size,
            component.eval.log_size(),
            component.claimed_sum,
        );
        let row_res = component.eval.evaluate(eval).row_res;

        // Finalize row.
        let row_denom_inv = denom_inv[row >> trace_log_size];
        res.set(row, accum.at(row) + row_res * row_denom_inv)
    }
    res
}
