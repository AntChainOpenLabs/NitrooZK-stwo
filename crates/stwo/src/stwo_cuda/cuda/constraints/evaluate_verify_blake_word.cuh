#ifndef EVALUATE_VERIFY_BLAKE_WORD_CONSTRAINT_H
#define EVALUATE_VERIFY_BLAKE_WORD_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "constraints/evaluate_mem_verify.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void verify_blake_word_evaluate(
    m31 verify_blake_word_input_limb_0,
    m31 verify_blake_word_input_limb_1,
    m31 verify_blake_word_input_limb_2,
    m31 low_7_ms_bits_col0,
    m31 high_14_ms_bits_col1,
    m31 high_5_ms_bits_col2,
    m31 id_col3,

    RangeCheck_7_2_5 range_check_7_2_5_lookup_elements,
    MemoryAddressToId memory_address_to_id_lookup_elements,
    MemoryIdToBig memory_id_to_big_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_0 = {0};
    const m31 M31_128 = {128};
    const m31 M31_4 = {4};
    const m31 M31_512 = {512};

    m31 values_0[3] = {
        low_7_ms_bits_col0,
        sub(verify_blake_word_input_limb_2, mul(high_14_ms_bits_col1, M31_4)),
        high_5_ms_bits_col2
    };
    RelationEntry entry_0 = RelationEntry<3>(range_check_7_2_5_lookup_elements, qm31{{1,0}, {0,0}}, values_0);
    cuda_evaluator->add_to_relation<3>(entry_0);

    mem_verify_evaluate (
        verify_blake_word_input_limb_0,
        sub(verify_blake_word_input_limb_1, mul(low_7_ms_bits_col0, M31_512)),
        add(low_7_ms_bits_col0, mul(sub(verify_blake_word_input_limb_2, mul(high_14_ms_bits_col1, M31_4)), M31_128)),
        sub(high_14_ms_bits_col1, mul(high_5_ms_bits_col2, M31_512)),
        high_5_ms_bits_col2,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        M31_0,
        id_col3,

        memory_address_to_id_lookup_elements,
        memory_id_to_big_lookup_elements,

        cuda_evaluator
    );

}


#endif // EVALUATE_VERIFY_BLAKE_WORD_CONSTRAINT_H