#ifndef DIV_252_H
#define DIV_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "add_252.cuh"  // For range_check_mem_value_n_28
#include "verify_mul_252.cuh"

// CUDA version Div252::evaluate
// translated from cairo-air/src/components/subroutines/div_252.rs
// 252-bit field division: (a / b) mod p, verified via result * b = a
//
// Structure (matching Rust Div252::evaluate):
// 1. Range check the result (not dividend!) using RangeCheckMemValueN28
// 2. Verify the multiplication: divisor * result = dividend using VerifyMul252

template<typename EvaluatorT>
DEVICE_FORCEINLINE void div_252_evaluate(
    const m31 input_a[28], // Dividend (what we're dividing)
    const m31 input_b[28], // Divisor (what we're dividing by)
    const m31 result[28],  // Result = input_a / input_b
    const m31 k,           // Quotient factor for verification
    const m31 carry[27],   // Carry values for verification
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
    const RangeCheck_19_G& rc_19_g,  // G is last
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check the RESULT (matching Rust Div252 line 92-131)
    // This ensures the division result is valid
    range_check_mem_value_n_28(
        result, rc_9_9, rc_9_9_b, rc_9_9_c, rc_9_9_d,
        rc_9_9_e, rc_9_9_f, rc_9_9_g, rc_9_9_h,
        cuda_evaluator
    );

    // Step 2: Verify division via multiplication: divisor * result = dividend
    // Rust Div252 calls VerifyMul252 with (input_a=divisor, div_res=result, input_c=dividend)
    // This verifies: divisor * result = dividend, i.e., result = dividend / divisor
    verify_mul_252_evaluate(
        input_b, result, input_a, k, carry,  // divisor * result = dividend
        rc_19_h, rc_19, rc_19_b, rc_19_c,
        rc_19_d, rc_19_e, rc_19_f, rc_19_g,
        cuda_evaluator
    );
}

#endif // DIV_252_H
