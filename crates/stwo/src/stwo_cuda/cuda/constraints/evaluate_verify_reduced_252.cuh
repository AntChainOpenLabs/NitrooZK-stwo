/*
============================================
VerifyReduced252 CUDA Subroutine
============================================

Subroutine: VerifyReduced252
translated from: cairo-air/src/comptogethernts/subroutines/verify_reduced_252.rs
AIR version: 54d95c0d

Functionality:
- Verifies that a 252-bit value is reduced (canonicalized) to the field's range
- Ensures the value is strictly less than the prime p (252-bit field modulus)
- Uses conditional constraints based on the most significant limbs

Inputs:
- verify_reduced_252_input_limb_0..27: 28 limbs of the 252-bit value (9 bits each)
- ms_limb_is_max: Boolean flag (1 if limb_27 is at maximum value)
- ms_and_mid_limbs_are_max: Boolean flag (1 if both MS and mid limbs are max)
- rc_input: Range check input value

Constraint Logic:
1. ms_limb_is_max is boolean: ms_limb_is_max * (1 - ms_limb_is_max) = 0
2. ms_and_mid_limbs_are_max is boolean: ms_and_mid_limbs_are_max * (1 - ms_and_mid_limbs_are_max) = 0
3. RangeCheck_8(limb_27 - ms_limb_is_max): Verify limb_27 < max if not flagged
4. If MS limb is max (ms_limb_is_max=1), then limbs 22-26 must be 0
5. rc_input = ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max)
6. RangeCheck_8(rc_input): Verify the computed range check input
7. If both MS and mid limbs are max (ms_and_mid_limbs_are_max=1), then limbs 0-20 must be 0

Relations:
- RangeCheck_8: 2 uses
  * (limb_27 - ms_limb_is_max)
  * rc_input

Constraint Count:
- 2 boolean constraints
- 5 conditional zero constraints (if MS is max)
- 1 rc_input constraint
- 21 conditional zero constraints (if both MS and mid are max)
- Total: 29 constraints

Key Algorithm:
This ensures that the 252-bit number is properly reduced modulo the field prime.
The constraints verify that the value doesn't exceed the field size by checking
the most significant limbs and ensuring proper bounds.
============================================
*/

#ifndef EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H
#define EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void verify_reduced_252_evaluate(
    m31 verify_reduced_252_input_limb_0,
    m31 verify_reduced_252_input_limb_1,
    m31 verify_reduced_252_input_limb_2,
    m31 verify_reduced_252_input_limb_3,
    m31 verify_reduced_252_input_limb_4,
    m31 verify_reduced_252_input_limb_5,
    m31 verify_reduced_252_input_limb_6,
    m31 verify_reduced_252_input_limb_7,
    m31 verify_reduced_252_input_limb_8,
    m31 verify_reduced_252_input_limb_9,
    m31 verify_reduced_252_input_limb_10,
    m31 verify_reduced_252_input_limb_11,
    m31 verify_reduced_252_input_limb_12,
    m31 verify_reduced_252_input_limb_13,
    m31 verify_reduced_252_input_limb_14,
    m31 verify_reduced_252_input_limb_15,
    m31 verify_reduced_252_input_limb_16,
    m31 verify_reduced_252_input_limb_17,
    m31 verify_reduced_252_input_limb_18,
    m31 verify_reduced_252_input_limb_19,
    m31 verify_reduced_252_input_limb_20,
    m31 verify_reduced_252_input_limb_21,
    m31 verify_reduced_252_input_limb_22,
    m31 verify_reduced_252_input_limb_23,
    m31 verify_reduced_252_input_limb_24,
    m31 verify_reduced_252_input_limb_25,
    m31 verify_reduced_252_input_limb_26,
    m31 verify_reduced_252_input_limb_27,

    m31 ms_limb_is_max_col0,
    m31 ms_and_mid_limbs_are_max_col1,
    m31 rc_input_col2,

    RangeCheck_8 range_check_8_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_1 = 1;
    const m31 M31_120 = 120;

    // ===================== Boolean Constraint 1: ms_limb_is_max is bit =====================
    // Constraint: ms_limb_is_max * (1 - ms_limb_is_max) = 0
    // This ensures ms_limb_is_max ∈ {0, 1}
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, sub(M31_1, ms_limb_is_max_col0))
    );

    // ===================== Boolean Constraint 2: ms_and_mid_limbs_are_max is bit =====================
    // Constraint: ms_and_mid_limbs_are_max * (1 - ms_and_mid_limbs_are_max) = 0
    // This ensures ms_and_mid_limbs_are_max ∈ {0, 1}
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, sub(M31_1, ms_and_mid_limbs_are_max_col1))
    );

    // ===================== RangeCheck_8: limb_27 - ms_limb_is_max =====================
    // Verify that limb_27 is within range when not flagged as max
    m31 rc_value_0[1] = {
        sub(verify_reduced_252_input_limb_27, ms_limb_is_max_col0)
    };
    RelationEntry<1> rc_entry_0(
        range_check_8_lookup_elements,
        qm31{{1, 0}, {0, 0}},  // positive multiplicity
        rc_value_0
    );
    cuda_evaluator->add_to_relation<1>(rc_entry_0);

    // ===================== If MS limb is max, high limbs (22-26) must be 0 =====================
    // These constraints ensure that if ms_limb_is_max = 1, then limbs 22-26 are all zero
    // Constraint: ms_limb_is_max * limb_22 = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, verify_reduced_252_input_limb_22)
    );

    // Constraint: ms_limb_is_max * limb_23 = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, verify_reduced_252_input_limb_23)
    );

    // Constraint: ms_limb_is_max * limb_24 = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, verify_reduced_252_input_limb_24)
    );

    // Constraint: ms_limb_is_max * limb_25 = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, verify_reduced_252_input_limb_25)
    );

    // Constraint: ms_limb_is_max * limb_26 = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, verify_reduced_252_input_limb_26)
    );

    // ===================== rc_input constraint =====================
    // rc_input = ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max)
    // Constraint: rc_input - ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max) = 0
    m31 expected_rc_input = mul(
        ms_limb_is_max_col0,
        sub(add(M31_120, verify_reduced_252_input_limb_21), ms_and_mid_limbs_are_max_col1)
    );
    cuda_evaluator->add_constraint(
        sub(rc_input_col2, expected_rc_input)
    );

    // ===================== RangeCheck_8: rc_input =====================
    m31 rc_value_1[1] = {rc_input_col2};
    RelationEntry<1> rc_entry_1(
        range_check_8_lookup_elements,
        qm31{{1, 0}, {0, 0}},  // positive multiplicity
        rc_value_1
    );
    cuda_evaluator->add_to_relation<1>(rc_entry_1);

    // ===================== If MS and mid limbs are max, low limbs (0-20) must be 0 =====================
    // These constraints ensure that if ms_and_mid_limbs_are_max = 1, then limbs 0-20 are all zero
    // Constraint: ms_and_mid_limbs_are_max * limb_0 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_0)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_1 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_1)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_2 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_2)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_3 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_3)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_4 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_4)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_5 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_5)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_6 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_6)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_7 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_7)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_8 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_8)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_9 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_9)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_10 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_10)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_11 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_11)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_12 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_12)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_13 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_13)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_14 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_14)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_15 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_15)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_16 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_16)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_17 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_17)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_18 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_18)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_19 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_19)
    );

    // Constraint: ms_and_mid_limbs_are_max * limb_20 = 0
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, verify_reduced_252_input_limb_20)
    );
}

#endif // EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H
