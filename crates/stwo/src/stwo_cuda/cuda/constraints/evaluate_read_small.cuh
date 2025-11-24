#ifndef EVALUATE_READ_SMALL_H
#define EVALUATE_READ_SMALL_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

DEVICE_FORCEINLINE void evaluate_cond_decode_small_sign(
    const m31 cond_decode_small_sign_input[29],
    m31 msb_col0,
    m31 mid_limbs_set_col1,
    m31* output_vec, // 2 elements
    struct CudaAssertEvaluator* cuda_evaluator
) {
    m31 M31_1 = {1};

    // msb is a bit.
    cuda_evaluator->add_constraint(
        mul(msb_col0, sub(msb_col0, M31_1))
    );
    // mid_limbs_set is a bit.
    cuda_evaluator->add_constraint(
        mul(mid_limbs_set_col1, sub(mid_limbs_set_col1, M31_1))
    );
    // Cannot have msb equals 0 and mid_limbs_set equals 1.
    cuda_evaluator->add_constraint(
        mul(
            mul(cond_decode_small_sign_input[28], mid_limbs_set_col1),
            sub(msb_col0, M31_1)
        )
    );

    output_vec[0] = msb_col0;
    output_vec[1] = mid_limbs_set_col1;
}


DEVICE_FORCEINLINE void evaluate_read_small(
    m31 read_small_input,
    m31 id_col0,
    m31 msb_col1,
    m31 mid_limbs_set_col2,
    m31 value_limb_0_col3,
    m31 value_limb_1_col4,
    m31 value_limb_2_col5,
    m31* output_vec, // 2 elements
    MemoryAddressToId memory_address_to_id_lookup_elements,
    MemoryIdToBig memory_id_to_big_lookup_elements,
    struct CudaAssertEvaluator* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_134217728 = {134217728};
    m31 M31_136 = {136};
    m31 M31_256 = {256};
    m31 M31_262144 = {262144};
    m31 M31_511 = {511};
    m31 M31_512 = {512};

    // memory_address_to_id lookup
    {
        m31 values[2] = { read_small_input, id_col0 };
        RelationEntry entry = RelationEntry<2>(
            memory_address_to_id_lookup_elements,
            qm31{{1,0},{0,0}},
            values
        );
        cuda_evaluator->add_to_relation<2>(entry);
    }

    // CondDecodeSmallSign
    m31 cond_decode_small_sign_input[29];
    for(int i = 0; i < 28; ++i) cond_decode_small_sign_input[i] = M31_0;
    cond_decode_small_sign_input[28] = M31_1;

    m31 cond_decode_small_sign_output[2];
    evaluate_cond_decode_small_sign(
        cond_decode_small_sign_input,
        msb_col1,
        mid_limbs_set_col2,
        cond_decode_small_sign_output,
        cuda_evaluator
    );
    // output: cond_decode_small_sign_output[0], cond_decode_small_sign_output[1]

    // memory_id_to_big lookup
    {
        m31 values[29] = {
            id_col0,
            value_limb_0_col3,
            value_limb_1_col4,
            value_limb_2_col5,
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            sub(mul(M31_136, msb_col1), mid_limbs_set_col2),
            M31_0,
            M31_0,
            M31_0,
            M31_0,
            M31_0,
            mul(msb_col1, M31_256)
        };
        RelationEntry entry = RelationEntry<29>(
            memory_id_to_big_lookup_elements,
            qm31{{1,0},{0,0}},
            values
        );
        cuda_evaluator->add_to_relation<29>(entry);
    }

    // Output
    output_vec[0] = sub(
        sub(
            add(
                add(
                    value_limb_0_col3,
                    mul(value_limb_1_col4, M31_512)
                ),
                mul(value_limb_2_col5, M31_262144)
            ),
            msb_col1
        ),
        mul(M31_134217728, mid_limbs_set_col2)
    );
    output_vec[1] = id_col0;
}


#endif // EVALUATE_READ_SMALL_H