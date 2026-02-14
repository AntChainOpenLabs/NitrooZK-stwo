#ifndef EVALUATE_RANGE_CHECK_LAST_LIMB_BITS_IN_MS_LIMB_H
#define EVALUATE_RANGE_CHECK_LAST_LIMB_BITS_IN_MS_LIMB_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_range_check_last_limb_bits_in_ms_limb_6(
    m31 range_check_last_limb_bits_in_ms_limb_6_input,
    EvaluatorT *cuda_evaluator,
    RangeCheck_6 range_check_6_lookup_elements
) {
    m31 values[1] = {range_check_last_limb_bits_in_ms_limb_6_input};
    RelationEntry<1> entry(
        range_check_6_lookup_elements,
        qm31{m31(1), m31(0)},
        values
    );
    cuda_evaluator->add_to_relation<1>(entry);
}


template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_range_check_last_limb_bits_in_ms_limb_2(
    m31 range_check_last_limb_bits_in_ms_limb_2_input,
    m31 msb_col0,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_1 = m31(1);
    m31 M31_2 = m31(2);

    // msb is a bit.
    cuda_evaluator->add_constraint(mul(msb_col0, sub(M31_1, msb_col0)));

    m31 bit_before_msb_tmp_f851f_1 = cuda_evaluator->add_intermediate(
        sub(range_check_last_limb_bits_in_ms_limb_2_input, mul(msb_col0, M31_2))
    );

    // bit before msb is a bit.
    cuda_evaluator->add_constraint(
        mul(bit_before_msb_tmp_f851f_1, sub(M31_1, bit_before_msb_tmp_f851f_1))
    );
}

#endif // EVALUATE_RANGE_CHECK_LAST_LIMB_BITS_IN_MS_LIMB_H
