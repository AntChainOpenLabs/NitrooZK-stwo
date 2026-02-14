use std::ops::Mul;

use hashbrown::HashMap;
use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::core::Fraction;
use stwo::prover::backend::simd::column::VeryPackedBaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField, LOG_N_VERY_PACKED_ELEMS,
};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::logup::LogupAtRow;
use crate::{Batching, EvalAtRow, INTERACTION_TRACE_IDX};

/// Evaluates constraints at an evaluation domain points.
pub struct SimdDomainEvaluator<'a> {
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: Vec<usize>,
    /// The row index of the simd-vector row to evaluate the constraints at.
    pub vec_row: usize,
    pub random_coeff_powers: &'a [SecureField],
    pub row_res: VeryPackedSecureField,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub logup: LogupAtRow<Self>,
}
impl<'a> SimdDomainEvaluator<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        random_coeff_powers: &'a [SecureField],
        domain_log_size: u32,
        eval_log_size: u32,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_eval,
            column_index_per_interaction: vec![0; trace_eval.len()],
            vec_row,
            random_coeff_powers,
            row_res: VeryPackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}
impl EvalAtRow for SimdDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    // TODO(Ohad): Add debug boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                unsafe {
                    let col = &self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index)
                        .values;
                    let very_packed_col = VeryPackedBaseColumn::transform_under_ref(col);
                    return *very_packed_col.data.get_unchecked(self.vec_row);
                };
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            VeryPackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS)) + i,
                    self.domain_log_size,
                    self.eval_domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(row_index)
            }))
        })
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        self.row_res +=
            VeryPackedSecureField::broadcast(self.random_coeff_powers[self.constraint_index])
                * constraint;
        self.constraint_index += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        VeryPackedSecureField::from_very_packed_m31s(values)
    }

    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        if self.logup.fracs.is_empty() {
            self.logup.is_finalized = false;
        }
        self.logup.fracs.push(fraction.clone());
    }

    fn finalize_logup_batched(&mut self, batching: &Batching) {
        assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");
        assert_eq!(
            batching.len(),
            self.logup.fracs.len(),
            "Batching must be of the same length as the number of entries"
        );

        let last_batch = *batching.iter().max().unwrap();

        let mut fracs_by_batch =
            HashMap::<usize, Vec<Fraction<Self::EF, Self::EF>>>::new();

        for (batch, frac) in batching.iter().zip(self.logup.fracs.iter()) {
            fracs_by_batch
                .entry(*batch)
                .or_insert_with(Vec::new)
                .push(frac.clone());
        }

        let keys_set: hashbrown::HashSet<_> = fracs_by_batch.keys().cloned().collect();
        let all_batches_set: hashbrown::HashSet<_> = (0..last_batch + 1).collect();

        assert_eq!(
            keys_set, all_batches_set,
            "Batching must contain all consecutive batches"
        );

        let mut prev_col_cumsum = <Self::EF as num_traits::Zero>::zero();

        // All batches except the last are cumulatively summed in new interaction columns.
        for batch_id in 0..last_batch {
            let cur_frac: Fraction<_, _> = fracs_by_batch[&batch_id].iter().cloned().sum();

            let [cur_cumsum] =
                self.next_extension_interaction_mask(self.logup.interaction, [0]);

            let diff = cur_cumsum.clone() - prev_col_cumsum.clone();
            prev_col_cumsum = cur_cumsum;
            let constraint_val = diff * cur_frac.denominator.clone() - cur_frac.numerator.clone();

            self.add_constraint(constraint_val);
        }

        let frac: Fraction<_, _> = fracs_by_batch[&last_batch].clone().into_iter().sum();
        let [prev_row_cumsum, cur_cumsum] =
            self.next_extension_interaction_mask(self.logup.interaction, [-1, 0]);

        let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum.clone();
        // Instead of checking diff = num / denom, check diff = num / denom - cumsum_shift.
        // This makes (num / denom - cumsum_shift) have sum zero, which makes the constraint
        // uniform - apply on all rows.
        let fixed_diff = diff + self.logup.cumsum_shift.clone();

        self.add_constraint(fixed_diff * frac.denominator - frac.numerator);

        self.logup.is_finalized = true;
    }

    fn finalize_logup(&mut self) {
        let batches = (0..self.logup.fracs.len()).collect();
        self.finalize_logup_batched(&batches)
    }

    fn finalize_logup_in_pairs(&mut self) {
        let batches = (0..self.logup.fracs.len()).map(|n| n / 2).collect();
        self.finalize_logup_batched(&batches)
    }
}
