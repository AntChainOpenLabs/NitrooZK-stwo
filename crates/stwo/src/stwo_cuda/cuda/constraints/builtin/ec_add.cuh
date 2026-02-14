#ifndef EC_ADD_H
#define EC_ADD_H

#include "fields.cuh"
#include "utils.cuh"
#include "add_252.cuh"
#include "sub_252.cuh"
#include "mul_252.cuh"
#include "div_252.cuh"

// CUDA version EcAdd::evaluate
// translated from cairo-air/src/components/subroutines/ec_add.rs
// Elliptic curve point addition: (x1,y1) + (x2,y2) = (x3,y3)
//
// Must match Rust sequence exactly:
// 1. Sub252(x2, x1) -> sub_res_0..27, sub_p_bit_28 (diff_x = x2 - x1)
// 2. Add252(x2, x1) -> add_res_29..56, sub_p_bit_57 (sum_x = x2 + x1)
// 3. Sub252(y2, y1) -> sub_res_58..85, sub_p_bit_86 (diff_y = y2 - y1)
// 4. Div252(diff_y, diff_x) -> slope = (y2-y1)/(x2-x1) (div_res_87..114)
// 5. Mul252(slope, slope) -> slope^2 (mul_res_143..170)
// 6. Sub252(slope^2, sum_x) -> x3 = slope^2 - (x1+x2) (sub_res_199..226)
// 7. Sub252(x1, x3) -> x1 - x3 (sub_res_228..255)
// 8. Mul252(slope, x1-x3) -> slope*(x1-x3) (mul_res_257..284)
// 9. Sub252(slope*(x1-x3), y1) -> y3 = slope*(x1-x3) - y1 (sub_res_313..340)

template<typename EvaluatorT>
DEVICE_FORCEINLINE void ec_add_evaluate(
    // Inputs: 112 limbs (x1[28], y1[28], x2[28], y2[28])
    const m31 x1[28], const m31 y1[28],
    const m31 x2[28], const m31 y2[28],
    // Outputs: trace columns for intermediate results (must match Rust column layout)
    m31 *sub_res_0,      // Step 1: x2 - x1 (28 limbs, cols 0-27)
    m31 *sub_p_bit_0,    // Step 1: borrow bit (col 28)
    m31 *add_res_0,      // Step 2: x2 + x1 (28 limbs, cols 29-56)
    m31 *sub_p_bit_1,    // Step 2: overflow bit (col 57)
    m31 *sub_res_1,      // Step 3: y2 - y1 (28 limbs, cols 58-85)
    m31 *sub_p_bit_2,    // Step 3: borrow bit (col 86)
    m31 *div_res,        // Step 4: slope = (y2-y1)/(x2-x1) (28 limbs, cols 87-114)
    m31 *k_div,          // Step 4: quotient (col 115)
    m31 *carry_div,      // Step 4: carries (27, cols 116-142)
    m31 *mul_res_0,      // Step 5: slope^2 (28 limbs, cols 143-170)
    m31 *k_mul_0,        // Step 5: quotient (col 171)
    m31 *carry_mul_0,    // Step 5: carries (27, cols 172-198)
    m31 *sub_res_2,      // Step 6: x3 = slope^2 - (x1+x2) (28 limbs, cols 199-226)
    m31 *sub_p_bit_3,    // Step 6: borrow bit (col 227)
    m31 *sub_res_3,      // Step 7: x1 - x3 (28 limbs, cols 228-255)
    m31 *sub_p_bit_4,    // Step 7: borrow bit (col 256)
    m31 *mul_res_1,      // Step 8: slope * (x1 - x3) (28 limbs, cols 257-284)
    m31 *k_mul_1,        // Step 8: quotient (col 285)
    m31 *carry_mul_1,    // Step 8: carries (27, cols 286-312)
    m31 *sub_res_4,      // Step 9: y3 = slope*(x1-x3) - y1 (28 limbs, cols 313-340)
    m31 *sub_p_bit_5,    // Step 9: borrow bit (col 341)
    // Range check lookup elements (order must match Rust EcAdd::evaluate)
    const RangeCheck_9_9& rc_9_9,
    const RangeCheck_9_9_B& rc_9_9_b,
    const RangeCheck_9_9_C& rc_9_9_c,
    const RangeCheck_9_9_D& rc_9_9_d,
    const RangeCheck_9_9_E& rc_9_9_e,
    const RangeCheck_9_9_F& rc_9_9_f,
    const RangeCheck_9_9_G& rc_9_9_g,
    const RangeCheck_9_9_H& rc_9_9_h,
    const RangeCheck_19_H& rc_19_h,  // H comes first for range_check_19!
    const RangeCheck_19& rc_19,
    const RangeCheck_19_B& rc_19_b,
    const RangeCheck_19_C& rc_19_c,
    const RangeCheck_19_D& rc_19_d,
    const RangeCheck_19_E& rc_19_e,
    const RangeCheck_19_F& rc_19_f,
    const RangeCheck_19_G& rc_19_g,  // G is last (no H at end)
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Compute diff_x = x2 - x1
    sub_252_evaluate(
        x2, x1, sub_res_0, *sub_p_bit_0,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 2: Compute sum_x = x2 + x1
    add_252_evaluate(
        x2, x1, add_res_0, *sub_p_bit_1,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 3: Compute diff_y = y2 - y1
    sub_252_evaluate(
        y2, y1, sub_res_1, *sub_p_bit_2,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 4: Compute slope = diff_y / diff_x = (y2 - y1) / (x2 - x1)
    div_252_evaluate(
        sub_res_1, sub_res_0, div_res, *k_div, carry_div,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        rc_19_h, rc_19, rc_19_b, rc_19_c,
        rc_19_d, rc_19_e, rc_19_f, rc_19_g,
        cuda_evaluator
    );

    // Step 5: Compute slope^2
    mul_252_evaluate(
        div_res, div_res, mul_res_0, *k_mul_0, carry_mul_0,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        rc_19_h, rc_19, rc_19_b, rc_19_c,
        rc_19_d, rc_19_e, rc_19_f, rc_19_g,
        cuda_evaluator
    );

    // Step 6: Compute x3 = slope^2 - sum_x = slope^2 - (x1 + x2)
    sub_252_evaluate(
        mul_res_0, add_res_0, sub_res_2, *sub_p_bit_3,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 7: Compute x1 - x3
    sub_252_evaluate(
        x1, sub_res_2, sub_res_3, *sub_p_bit_4,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 8: Compute slope * (x1 - x3)
    mul_252_evaluate(
        div_res, sub_res_3, mul_res_1, *k_mul_1, carry_mul_1,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        rc_19_h, rc_19, rc_19_b, rc_19_c,
        rc_19_d, rc_19_e, rc_19_f, rc_19_g,
        cuda_evaluator
    );

    // Step 9: Compute y3 = slope * (x1 - x3) - y1
    sub_252_evaluate(
        mul_res_1, y1, sub_res_4, *sub_p_bit_5,
        rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Result: sub_res_2 = x3, sub_res_4 = y3
}

#endif // EC_ADD_H
