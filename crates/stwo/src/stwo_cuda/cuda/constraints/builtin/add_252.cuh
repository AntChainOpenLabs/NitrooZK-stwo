#ifndef ADD_252_H
#define ADD_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "../evaluate_verify_add_252.cuh"

// CUDAversion Add252::evaluate
// translated from cairo-air/src/comptogethernts/subroutines/add_252.rs
// 252-bitfieldaddition: (a + b) mod p，p = 2^252 + 17*2^192 + 1

// Range check helper for 28 limbs using 8 RangeCheck_9_9 variants
template<typename EvaluatorT>
DEVICE_FORCEINLINE void range_check_mem_value_n_28(
    const m31 limbs[28],  // 28 limbs to range check
    const RangeCheck_9_9& rc_9_9,
    const RangeCheck_9_9_B& rc_9_9_b,
    const RangeCheck_9_9_C& rc_9_9_c,
    const RangeCheck_9_9_D& rc_9_9_d,
    const RangeCheck_9_9_E& rc_9_9_e,
    const RangeCheck_9_9_F& rc_9_9_f,
    const RangeCheck_9_9_G& rc_9_9_g,
    const RangeCheck_9_9_H& rc_9_9_h,
    EvaluatorT* cuda_evaluator
) {
    // Check limbs in pairs using 8 different RangeCheck_9_9 variants
    // Pattern: A,B,C,D,E,F,G,H repeats for 28 limbs (14 pairs)

    // Pairs 0-1: RangeCheck_9_9
    m31 values0[2] = {limbs[0], limbs[1]};
    RelationEntry<2> entry0(rc_9_9, qm31{{1, 0}, {0, 0}}, values0);
    cuda_evaluator->template add_to_relation<2>(entry0);

    // Pairs 2-3: RangeCheck_9_9_B
    m31 values1[2] = {limbs[2], limbs[3]};
    RelationEntry<2> entry1(rc_9_9_b, qm31{{1, 0}, {0, 0}}, values1);
    cuda_evaluator->template add_to_relation<2>(entry1);

    // Pairs 4-5: RangeCheck_9_9_C
    m31 values2[2] = {limbs[4], limbs[5]};
    RelationEntry<2> entry2(rc_9_9_c, qm31{{1, 0}, {0, 0}}, values2);
    cuda_evaluator->template add_to_relation<2>(entry2);

    // Pairs 6-7: RangeCheck_9_9_D
    m31 values3[2] = {limbs[6], limbs[7]};
    RelationEntry<2> entry3(rc_9_9_d, qm31{{1, 0}, {0, 0}}, values3);
    cuda_evaluator->template add_to_relation<2>(entry3);

    // Pairs 8-9: RangeCheck_9_9_E
    m31 values4[2] = {limbs[8], limbs[9]};
    RelationEntry<2> entry4(rc_9_9_e, qm31{{1, 0}, {0, 0}}, values4);
    cuda_evaluator->template add_to_relation<2>(entry4);

    // Pairs 10-11: RangeCheck_9_9_F
    m31 values5[2] = {limbs[10], limbs[11]};
    RelationEntry<2> entry5(rc_9_9_f, qm31{{1, 0}, {0, 0}}, values5);
    cuda_evaluator->template add_to_relation<2>(entry5);

    // Pairs 12-13: RangeCheck_9_9_G
    m31 values6[2] = {limbs[12], limbs[13]};
    RelationEntry<2> entry6(rc_9_9_g, qm31{{1, 0}, {0, 0}}, values6);
    cuda_evaluator->template add_to_relation<2>(entry6);

    // Pairs 14-15: RangeCheck_9_9_H
    m31 values7[2] = {limbs[14], limbs[15]};
    RelationEntry<2> entry7(rc_9_9_h, qm31{{1, 0}, {0, 0}}, values7);
    cuda_evaluator->template add_to_relation<2>(entry7);

    // Repeat pattern for remaining limbs
    // Pairs 16-17: RangeCheck_9_9
    m31 values8[2] = {limbs[16], limbs[17]};
    RelationEntry<2> entry8(rc_9_9, qm31{{1, 0}, {0, 0}}, values8);
    cuda_evaluator->template add_to_relation<2>(entry8);

    // Pairs 18-19: RangeCheck_9_9_B
    m31 values9[2] = {limbs[18], limbs[19]};
    RelationEntry<2> entry9(rc_9_9_b, qm31{{1, 0}, {0, 0}}, values9);
    cuda_evaluator->template add_to_relation<2>(entry9);

    // Pairs 20-21: RangeCheck_9_9_C
    m31 values10[2] = {limbs[20], limbs[21]};
    RelationEntry<2> entry10(rc_9_9_c, qm31{{1, 0}, {0, 0}}, values10);
    cuda_evaluator->template add_to_relation<2>(entry10);

    // Pairs 22-23: RangeCheck_9_9_D
    m31 values11[2] = {limbs[22], limbs[23]};
    RelationEntry<2> entry11(rc_9_9_d, qm31{{1, 0}, {0, 0}}, values11);
    cuda_evaluator->template add_to_relation<2>(entry11);

    // Pairs 24-25: RangeCheck_9_9_E
    m31 values12[2] = {limbs[24], limbs[25]};
    RelationEntry<2> entry12(rc_9_9_e, qm31{{1, 0}, {0, 0}}, values12);
    cuda_evaluator->template add_to_relation<2>(entry12);

    // Pairs 26-27: RangeCheck_9_9_F
    m31 values13[2] = {limbs[26], limbs[27]};
    RelationEntry<2> entry13(rc_9_9_f, qm31{{1, 0}, {0, 0}}, values13);
    cuda_evaluator->template add_to_relation<2>(entry13);
}

// Main Add252 function
// Computes: result = (a + b) mod p where p = 2^252 + 17*2^192 + 1
template<typename EvaluatorT>
DEVICE_FORCEINLINE void add_252_evaluate(
    const m31 input_a[28],   // First operand (28 limbs)
    const m31 input_b[28],   // Second operand (28 limbs)
    const m31 result[28],    // Result of addition (28 limbs)
    const m31 sub_p_bit,     // Boolean: whether we subtracted p
    const RangeCheck_9_9& rc_9_9,
    const RangeCheck_9_9_B& rc_9_9_b,
    const RangeCheck_9_9_C& rc_9_9_c,
    const RangeCheck_9_9_D& rc_9_9_d,
    const RangeCheck_9_9_E& rc_9_9_e,
    const RangeCheck_9_9_F& rc_9_9_f,
    const RangeCheck_9_9_G& rc_9_9_g,
    const RangeCheck_9_9_H& rc_9_9_h,
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check the result limbs
    range_check_mem_value_n_28(
        result, rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 2: Verify the addition is correct
    // This checks: a + b = result + sub_p_bit * p
    evaluate_verify_add_252(
        input_a[0], input_a[1], input_a[2], input_a[3], input_a[4], input_a[5], input_a[6], input_a[7],
        input_a[8], input_a[9], input_a[10], input_a[11], input_a[12], input_a[13], input_a[14], input_a[15],
        input_a[16], input_a[17], input_a[18], input_a[19], input_a[20], input_a[21], input_a[22], input_a[23],
        input_a[24], input_a[25], input_a[26], input_a[27],
        input_b[0], input_b[1], input_b[2], input_b[3], input_b[4], input_b[5], input_b[6], input_b[7],
        input_b[8], input_b[9], input_b[10], input_b[11], input_b[12], input_b[13], input_b[14], input_b[15],
        input_b[16], input_b[17], input_b[18], input_b[19], input_b[20], input_b[21], input_b[22], input_b[23],
        input_b[24], input_b[25], input_b[26], input_b[27],
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15],
        result[16], result[17], result[18], result[19], result[20], result[21], result[22], result[23],
        result[24], result[25], result[26], result[27],
        sub_p_bit,
        cuda_evaluator
    );
}

#endif // ADD_252_H
