// mul_opcode CUDA AIR Evaluator
// This evaluator handles the multiplication opcode constraints for Cairo VM
// 130 trace columns, complex 252-bit multiplication verification with Karatsuba algorithm

#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_mul_opcode.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_common.cuh"

// M31 constants used in constraints
__device__ __constant__ m31 M31_1 = {1};
__device__ __constant__ m31 M31_2 = {2};
__device__ __constant__ m31 M31_4 = {4};
__device__ __constant__ m31 M31_8 = {8};
__device__ __constant__ m31 M31_32 = {32};
__device__ __constant__ m31 M31_64 = {64};
__device__ __constant__ m31 M31_136 = {136};
__device__ __constant__ m31 M31_256 = {256};
__device__ __constant__ m31 M31_512 = {512};
__device__ __constant__ m31 M31_131072 = {131072};
__device__ __constant__ m31 M31_262144 = {262144};

// =====================================================================
// Helper: SingleKaratsubaN7
// Performs Karatsuba multiplication of two 14-limb numbers
// Input: 28 limbs (x[0..14] and y[0..14], where y = input[14..28])
// Output: 27 limbs
// Matches Rust's single_karatsuba_n_7.rs polynomial structure exactly
// =====================================================================
template<typename EvaluatorT>
__device__ __forceinline__ void SingleKaratsubaN7(
    EvaluatorT* eval,
    m31 input[28],  // x[0..14] in input[0..14], y[0..14] in input[14..28]
    m31 output[27]  // Output (27 limbs)
) {
    // Split input: x = input[0..14], y = input[14..28]
    // Further split: x_lo = x[0..7], x_hi = x[7..14]
    //                y_lo = y[0..7], y_hi = y[7..14]

    // z0 = x_lo * y_lo (schoolbook, 13 limbs)
    // Using add_intermediate equivalent structure
    m31 z0[13];

    // z0_limb_0 = x[0] * y[0]
    z0[0] = mul(input[0], input[14]);

    // z0_limb_1 = x[0]*y[1] + x[1]*y[0]
    z0[1] = add(mul(input[0], input[15]), mul(input[1], input[14]));

    // z0_limb_2 = x[0]*y[2] + x[1]*y[1] + x[2]*y[0]
    z0[2] = add(add(mul(input[0], input[16]), mul(input[1], input[15])), mul(input[2], input[14]));

    // z0_limb_3
    z0[3] = add(add(add(mul(input[0], input[17]), mul(input[1], input[16])), mul(input[2], input[15])), mul(input[3], input[14]));

    // z0_limb_4
    z0[4] = add(add(add(add(mul(input[0], input[18]), mul(input[1], input[17])), mul(input[2], input[16])), mul(input[3], input[15])), mul(input[4], input[14]));

    // z0_limb_5
    z0[5] = add(add(add(add(add(mul(input[0], input[19]), mul(input[1], input[18])), mul(input[2], input[17])), mul(input[3], input[16])), mul(input[4], input[15])), mul(input[5], input[14]));

    // z0_limb_6
    z0[6] = add(add(add(add(add(add(mul(input[0], input[20]), mul(input[1], input[19])), mul(input[2], input[18])), mul(input[3], input[17])), mul(input[4], input[16])), mul(input[5], input[15])), mul(input[6], input[14]));

    // z0_limb_7 (no x[0] term)
    z0[7] = add(add(add(add(add(mul(input[1], input[20]), mul(input[2], input[19])), mul(input[3], input[18])), mul(input[4], input[17])), mul(input[5], input[16])), mul(input[6], input[15]));

    // z0_limb_8
    z0[8] = add(add(add(add(mul(input[2], input[20]), mul(input[3], input[19])), mul(input[4], input[18])), mul(input[5], input[17])), mul(input[6], input[16]));

    // z0_limb_9
    z0[9] = add(add(add(mul(input[3], input[20]), mul(input[4], input[19])), mul(input[5], input[18])), mul(input[6], input[17]));

    // z0_limb_10
    z0[10] = add(add(mul(input[4], input[20]), mul(input[5], input[19])), mul(input[6], input[18]));

    // z0_limb_11
    z0[11] = add(mul(input[5], input[20]), mul(input[6], input[19]));

    // z0_limb_12
    z0[12] = mul(input[6], input[20]);

    // z2 = x_hi * y_hi (schoolbook, 13 limbs)
    // x_hi = input[7..14], y_hi = input[21..28]
    m31 z2[13];

    // z2_limb_0 = x[7] * y[7]
    z2[0] = mul(input[7], input[21]);

    // z2_limb_1
    z2[1] = add(mul(input[7], input[22]), mul(input[8], input[21]));

    // z2_limb_2
    z2[2] = add(add(mul(input[7], input[23]), mul(input[8], input[22])), mul(input[9], input[21]));

    // z2_limb_3
    z2[3] = add(add(add(mul(input[7], input[24]), mul(input[8], input[23])), mul(input[9], input[22])), mul(input[10], input[21]));

    // z2_limb_4
    z2[4] = add(add(add(add(mul(input[7], input[25]), mul(input[8], input[24])), mul(input[9], input[23])), mul(input[10], input[22])), mul(input[11], input[21]));

    // z2_limb_5
    z2[5] = add(add(add(add(add(mul(input[7], input[26]), mul(input[8], input[25])), mul(input[9], input[24])), mul(input[10], input[23])), mul(input[11], input[22])), mul(input[12], input[21]));

    // z2_limb_6
    z2[6] = add(add(add(add(add(add(mul(input[7], input[27]), mul(input[8], input[26])), mul(input[9], input[25])), mul(input[10], input[24])), mul(input[11], input[23])), mul(input[12], input[22])), mul(input[13], input[21]));

    // z2_limb_7
    z2[7] = add(add(add(add(add(mul(input[8], input[27]), mul(input[9], input[26])), mul(input[10], input[25])), mul(input[11], input[24])), mul(input[12], input[23])), mul(input[13], input[22]));

    // z2_limb_8
    z2[8] = add(add(add(add(mul(input[9], input[27]), mul(input[10], input[26])), mul(input[11], input[25])), mul(input[12], input[24])), mul(input[13], input[23]));

    // z2_limb_9
    z2[9] = add(add(add(mul(input[10], input[27]), mul(input[11], input[26])), mul(input[12], input[25])), mul(input[13], input[24]));

    // z2_limb_10
    z2[10] = add(add(mul(input[11], input[27]), mul(input[12], input[26])), mul(input[13], input[25]));

    // z2_limb_11
    z2[11] = add(mul(input[12], input[27]), mul(input[13], input[26]));

    // z2_limb_12
    z2[12] = mul(input[13], input[27]);

    // x_sum = x_lo + x_hi (7 limbs)
    m31 x_sum[7];
    x_sum[0] = add(input[0], input[7]);
    x_sum[1] = add(input[1], input[8]);
    x_sum[2] = add(input[2], input[9]);
    x_sum[3] = add(input[3], input[10]);
    x_sum[4] = add(input[4], input[11]);
    x_sum[5] = add(input[5], input[12]);
    x_sum[6] = add(input[6], input[13]);

    // y_sum = y_lo + y_hi (7 limbs)
    m31 y_sum[7];
    y_sum[0] = add(input[14], input[21]);
    y_sum[1] = add(input[15], input[22]);
    y_sum[2] = add(input[16], input[23]);
    y_sum[3] = add(input[17], input[24]);
    y_sum[4] = add(input[18], input[25]);
    y_sum[5] = add(input[19], input[26]);
    y_sum[6] = add(input[20], input[27]);

    // Output assembly using Karatsuba formula:
    // output[0..7] = z0[0..7]
    // output[7..14] = z0[7..13] + (x_sum*y_sum - z0 - z2)[0..7]
    // output[13] = (x_sum*y_sum - z0 - z2)[6] (no z0 term)
    // output[14..21] = z2[0..7] + (x_sum*y_sum - z0 - z2)[7..13]
    // output[20..27] = z2[6..13]

    // output[0..7] = z0[0..7]
    output[0] = z0[0];
    output[1] = z0[1];
    output[2] = z0[2];
    output[3] = z0[3];
    output[4] = z0[4];
    output[5] = z0[5];
    output[6] = z0[6];

    // output[7] = z0[7] + (x_sum*y_sum - z0 - z2)[0]
    // (x_sum*y_sum)[0] = x_sum[0]*y_sum[0]
    output[7] = add(z0[7], sub(sub(mul(x_sum[0], y_sum[0]), z0[0]), z2[0]));

    // output[8] = z0[8] + (x_sum*y_sum - z0 - z2)[1]
    output[8] = add(z0[8], sub(sub(add(mul(x_sum[0], y_sum[1]), mul(x_sum[1], y_sum[0])), z0[1]), z2[1]));

    // output[9]
    output[9] = add(z0[9], sub(sub(add(add(mul(x_sum[0], y_sum[2]), mul(x_sum[1], y_sum[1])), mul(x_sum[2], y_sum[0])), z0[2]), z2[2]));

    // output[10]
    output[10] = add(z0[10], sub(sub(add(add(add(mul(x_sum[0], y_sum[3]), mul(x_sum[1], y_sum[2])), mul(x_sum[2], y_sum[1])), mul(x_sum[3], y_sum[0])), z0[3]), z2[3]));

    // output[11]
    output[11] = add(z0[11], sub(sub(add(add(add(add(mul(x_sum[0], y_sum[4]), mul(x_sum[1], y_sum[3])), mul(x_sum[2], y_sum[2])), mul(x_sum[3], y_sum[1])), mul(x_sum[4], y_sum[0])), z0[4]), z2[4]));

    // output[12]
    output[12] = add(z0[12], sub(sub(add(add(add(add(add(mul(x_sum[0], y_sum[5]), mul(x_sum[1], y_sum[4])), mul(x_sum[2], y_sum[3])), mul(x_sum[3], y_sum[2])), mul(x_sum[4], y_sum[1])), mul(x_sum[5], y_sum[0])), z0[5]), z2[5]));

    // output[13] - this is the middle term with no z0 component
    output[13] = sub(sub(add(add(add(add(add(add(mul(x_sum[0], y_sum[6]), mul(x_sum[1], y_sum[5])), mul(x_sum[2], y_sum[4])), mul(x_sum[3], y_sum[3])), mul(x_sum[4], y_sum[2])), mul(x_sum[5], y_sum[1])), mul(x_sum[6], y_sum[0])), z0[6]), z2[6]);

    // output[14] = z2[0] + (x_sum*y_sum - z0 - z2)[7]
    output[14] = add(z2[0], sub(sub(add(add(add(add(add(mul(x_sum[1], y_sum[6]), mul(x_sum[2], y_sum[5])), mul(x_sum[3], y_sum[4])), mul(x_sum[4], y_sum[3])), mul(x_sum[5], y_sum[2])), mul(x_sum[6], y_sum[1])), z0[7]), z2[7]));

    // output[15]
    output[15] = add(z2[1], sub(sub(add(add(add(add(mul(x_sum[2], y_sum[6]), mul(x_sum[3], y_sum[5])), mul(x_sum[4], y_sum[4])), mul(x_sum[5], y_sum[3])), mul(x_sum[6], y_sum[2])), z0[8]), z2[8]));

    // output[16]
    output[16] = add(z2[2], sub(sub(add(add(add(mul(x_sum[3], y_sum[6]), mul(x_sum[4], y_sum[5])), mul(x_sum[5], y_sum[4])), mul(x_sum[6], y_sum[3])), z0[9]), z2[9]));

    // output[17]
    output[17] = add(z2[3], sub(sub(add(add(mul(x_sum[4], y_sum[6]), mul(x_sum[5], y_sum[5])), mul(x_sum[6], y_sum[4])), z0[10]), z2[10]));

    // output[18]
    output[18] = add(z2[4], sub(sub(add(mul(x_sum[5], y_sum[6]), mul(x_sum[6], y_sum[5])), z0[11]), z2[11]));

    // output[19]
    output[19] = add(z2[5], sub(sub(mul(x_sum[6], y_sum[6]), z0[12]), z2[12]));

    // output[20..27] = z2[6..13]
    output[20] = z2[6];
    output[21] = z2[7];
    output[22] = z2[8];
    output[23] = z2[9];
    output[24] = z2[10];
    output[25] = z2[11];
    output[26] = z2[12];
}

// =====================================================================
// Helper: DoubleKaratsubaN7LimbMaxBound511
// Performs Karatsuba multiplication of two 28-limb numbers (252 bits)
// Returns 55 limbs (double precision result)
// Matches Rust's double_karatsuba_n_7_limb_max_bound_511.rs exactly
// =====================================================================
template<typename EvaluatorT>
__device__ __forceinline__ void DoubleKaratsubaN7LimbMaxBound511(
    EvaluatorT* eval,
    m31 a[28],  // Input A (28 limbs, 9 bits each)
    m31 b[28],  // Input B (28 limbs, 9 bits each)
    m31 result[55]  // Output (55 limbs)
) {
    // Input mapping from Rust:
    // a = input[0..28] (limbs 0-27)
    // b = input[28..56] (limbs 28-55)

    // First SingleKaratsubaN7: inputs [0..14] and [28..42]
    // i.e., a[0..14] and b[0..14]
    m31 sk1_input[28];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        sk1_input[i] = a[i];
        sk1_input[i + 14] = b[i];
    }
    m31 z0_result[27];
    SingleKaratsubaN7<EvaluatorT>(eval, sk1_input, z0_result);

    // Second SingleKaratsubaN7: inputs [14..28] and [42..56]
    // i.e., a[14..28] and b[14..28]
    m31 sk2_input[28];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        sk2_input[i] = a[i + 14];
        sk2_input[i + 14] = b[i + 14];
    }
    m31 z2_result[27];
    SingleKaratsubaN7<EvaluatorT>(eval, sk2_input, z2_result);

    // Compute x_sum = a[0..14] + a[14..28]
    m31 x_sum[14];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        x_sum[i] = add(a[i], a[i + 14]);
    }

    // Compute y_sum = b[0..14] + b[14..28]
    m31 y_sum[14];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        y_sum[i] = add(b[i], b[i + 14]);
    }

    // Third SingleKaratsubaN7: x_sum * y_sum
    m31 sk3_input[28];
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        sk3_input[i] = x_sum[i];
        sk3_input[i + 14] = y_sum[i];
    }
    m31 z1_result[27];
    SingleKaratsubaN7<EvaluatorT>(eval, sk3_input, z1_result);

    // Assemble output using Karatsuba reconstruction:
    // result[0..14] = z0[0..14]
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        result[i] = z0_result[i];
    }

    // result[14..27] = z0[14..27] + (z1 - z0 - z2)[0..13]
    #pragma unroll
    for (int i = 0; i < 13; i++) {
        result[i + 14] = add(z0_result[i + 14], sub(sub(z1_result[i], z0_result[i]), z2_result[i]));
    }

    // result[27] = (z1 - z0 - z2)[13] (no z0 component)
    result[27] = sub(sub(z1_result[13], z0_result[13]), z2_result[13]);

    // result[28..41] = z2[0..13] + (z1 - z0 - z2)[14..27]
    #pragma unroll
    for (int i = 0; i < 13; i++) {
        result[i + 28] = add(z2_result[i], sub(sub(z1_result[i + 14], z0_result[i + 14]), z2_result[i + 14]));
    }

    // result[41..55] = z2[13..27]
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        result[i + 41] = z2_result[i + 13];
    }
}

// =====================================================================
// Helper: VerifyMul252
// Verifies that op0 * op1 = dst (mod Cairo prime)
// Cairo prime = 2^251 + 17 * 2^192 + 1
// Uses modular reduction and carry propagation
// =====================================================================
template<typename EvaluatorT>
__device__ __forceinline__ void VerifyMul252(
    EvaluatorT* eval,
    m31 op0_limbs[28],
    m31 op1_limbs[28],
    m31 dst_limbs[28],
    m31 k,
    m31 carries[27],
    MulOpcode_Eval* mul_eval
) {
    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);

    // Step 1: Compute op0 * op1 using DoubleKaratsuba
    m31 mul_result[55];
    DoubleKaratsubaN7LimbMaxBound511(eval, op0_limbs, op1_limbs, mul_result);

    // Step 2: Compute conv_tmp = mul_result - dst (first 28 limbs)
    //         and just mul_result for limbs 28-54
    m31 conv_tmp[55];
    #pragma unroll
    for (int i = 0; i < 28; i++) {
        conv_tmp[i] = sub(mul_result[i], dst_limbs[i]);
    }
    #pragma unroll
    for (int i = 28; i < 55; i++) {
        conv_tmp[i] = mul_result[i];
    }

    // Step 3: Apply modular reduction for Cairo prime p = 2^251 + 17*2^192 + 1
    // This is done by reducing high limbs using the relation:
    // 2^252 ≡ 2^252 - p * 2 = -2*17*2^192 - 2 = -34*2^192 - 2 (mod p)
    // Which gives: 2^252 ≡ -32*2^192 - 4*2^21 + 8*2^49 (simplified modulo arithmetic)
    //
    // The reduction formula for each limb is complex and follows the pattern
    // in verify_mul_252.rs lines 321-485

    m31 conv_mod[28];

    // Limb 0: 32*conv[0] - 4*conv[21] + 8*conv[49]
    conv_mod[0] = sub(add(mul(m31(32), conv_tmp[0]), mul(m31(8), conv_tmp[49])), mul(m31(4), conv_tmp[21]));

    // Limb 1: conv[0] + 32*conv[1] - 4*conv[22] + 8*conv[50]
    conv_mod[1] = sub(add(add(conv_tmp[0], mul(m31(32), conv_tmp[1])), mul(m31(8), conv_tmp[50])), mul(m31(4), conv_tmp[22]));

    // Limb 2: conv[1] + 32*conv[2] - 4*conv[23] + 8*conv[51]
    conv_mod[2] = sub(add(add(conv_tmp[1], mul(m31(32), conv_tmp[2])), mul(m31(8), conv_tmp[51])), mul(m31(4), conv_tmp[23]));

    // Limb 3: conv[2] + 32*conv[3] - 4*conv[24] + 8*conv[52]
    conv_mod[3] = sub(add(add(conv_tmp[2], mul(m31(32), conv_tmp[3])), mul(m31(8), conv_tmp[52])), mul(m31(4), conv_tmp[24]));

    // Limb 4: conv[3] + 32*conv[4] - 4*conv[25] + 8*conv[53]
    conv_mod[4] = sub(add(add(conv_tmp[3], mul(m31(32), conv_tmp[4])), mul(m31(8), conv_tmp[53])), mul(m31(4), conv_tmp[25]));

    // Limb 5: conv[4] + 32*conv[5] - 4*conv[26] + 8*conv[54]
    conv_mod[5] = sub(add(add(conv_tmp[4], mul(m31(32), conv_tmp[5])), mul(m31(8), conv_tmp[54])), mul(m31(4), conv_tmp[26]));

    // Limb 6: conv[5] + 32*conv[6] - 4*conv[27]
    conv_mod[6] = sub(add(conv_tmp[5], mul(m31(32), conv_tmp[6])), mul(m31(4), conv_tmp[27]));

    // Limb 7: 2*conv[0] + conv[6] + 32*conv[7] - 4*conv[28]
    conv_mod[7] = sub(add(add(mul(m31(2), conv_tmp[0]), conv_tmp[6]), mul(m31(32), conv_tmp[7])), mul(m31(4), conv_tmp[28]));

    // Limb 8: 2*conv[1] + conv[7] + 32*conv[8] - 4*conv[29]
    conv_mod[8] = sub(add(add(mul(m31(2), conv_tmp[1]), conv_tmp[7]), mul(m31(32), conv_tmp[8])), mul(m31(4), conv_tmp[29]));

    // Limb 9: 2*conv[2] + conv[8] + 32*conv[9] - 4*conv[30]
    conv_mod[9] = sub(add(add(mul(m31(2), conv_tmp[2]), conv_tmp[8]), mul(m31(32), conv_tmp[9])), mul(m31(4), conv_tmp[30]));

    // Limb 10: 2*conv[3] + conv[9] + 32*conv[10] - 4*conv[31]
    conv_mod[10] = sub(add(add(mul(m31(2), conv_tmp[3]), conv_tmp[9]), mul(m31(32), conv_tmp[10])), mul(m31(4), conv_tmp[31]));

    // Limb 11: 2*conv[4] + conv[10] + 32*conv[11] - 4*conv[32]
    conv_mod[11] = sub(add(add(mul(m31(2), conv_tmp[4]), conv_tmp[10]), mul(m31(32), conv_tmp[11])), mul(m31(4), conv_tmp[32]));

    // Limb 12: 2*conv[5] + conv[11] + 32*conv[12] - 4*conv[33]
    conv_mod[12] = sub(add(add(mul(m31(2), conv_tmp[5]), conv_tmp[11]), mul(m31(32), conv_tmp[12])), mul(m31(4), conv_tmp[33]));

    // Limb 13: 2*conv[6] + conv[12] + 32*conv[13] - 4*conv[34]
    conv_mod[13] = sub(add(add(mul(m31(2), conv_tmp[6]), conv_tmp[12]), mul(m31(32), conv_tmp[13])), mul(m31(4), conv_tmp[34]));

    // Limb 14: 2*conv[7] + conv[13] + 32*conv[14] - 4*conv[35]
    conv_mod[14] = sub(add(add(mul(m31(2), conv_tmp[7]), conv_tmp[13]), mul(m31(32), conv_tmp[14])), mul(m31(4), conv_tmp[35]));

    // Limb 15: 2*conv[8] + conv[14] + 32*conv[15] - 4*conv[36]
    conv_mod[15] = sub(add(add(mul(m31(2), conv_tmp[8]), conv_tmp[14]), mul(m31(32), conv_tmp[15])), mul(m31(4), conv_tmp[36]));

    // Limb 16: 2*conv[9] + conv[15] + 32*conv[16] - 4*conv[37]
    conv_mod[16] = sub(add(add(mul(m31(2), conv_tmp[9]), conv_tmp[15]), mul(m31(32), conv_tmp[16])), mul(m31(4), conv_tmp[37]));

    // Limb 17: 2*conv[10] + conv[16] + 32*conv[17] - 4*conv[38]
    conv_mod[17] = sub(add(add(mul(m31(2), conv_tmp[10]), conv_tmp[16]), mul(m31(32), conv_tmp[17])), mul(m31(4), conv_tmp[38]));

    // Limb 18: 2*conv[11] + conv[17] + 32*conv[18] - 4*conv[39]
    conv_mod[18] = sub(add(add(mul(m31(2), conv_tmp[11]), conv_tmp[17]), mul(m31(32), conv_tmp[18])), mul(m31(4), conv_tmp[39]));

    // Limb 19: 2*conv[12] + conv[18] + 32*conv[19] - 4*conv[40]
    conv_mod[19] = sub(add(add(mul(m31(2), conv_tmp[12]), conv_tmp[18]), mul(m31(32), conv_tmp[19])), mul(m31(4), conv_tmp[40]));

    // Limb 20: 2*conv[13] + conv[19] + 32*conv[20] - 4*conv[41]
    conv_mod[20] = sub(add(add(mul(m31(2), conv_tmp[13]), conv_tmp[19]), mul(m31(32), conv_tmp[20])), mul(m31(4), conv_tmp[41]));

    // Limb 21: 2*conv[14] + conv[20] - 4*conv[42] + 64*conv[49]
    conv_mod[21] = sub(add(add(mul(m31(2), conv_tmp[14]), conv_tmp[20]), mul(m31(64), conv_tmp[49])), mul(m31(4), conv_tmp[42]));

    // Limb 22: 2*conv[15] - 4*conv[43] + 2*conv[49] + 64*conv[50]
    conv_mod[22] = add(add(sub(mul(m31(2), conv_tmp[15]), mul(m31(4), conv_tmp[43])), mul(m31(2), conv_tmp[49])), mul(m31(64), conv_tmp[50]));

    // Limb 23: 2*conv[16] - 4*conv[44] + 2*conv[50] + 64*conv[51]
    conv_mod[23] = add(add(sub(mul(m31(2), conv_tmp[16]), mul(m31(4), conv_tmp[44])), mul(m31(2), conv_tmp[50])), mul(m31(64), conv_tmp[51]));

    // Limb 24: 2*conv[17] - 4*conv[45] + 2*conv[51] + 64*conv[52]
    conv_mod[24] = add(add(sub(mul(m31(2), conv_tmp[17]), mul(m31(4), conv_tmp[45])), mul(m31(2), conv_tmp[51])), mul(m31(64), conv_tmp[52]));

    // Limb 25: 2*conv[18] - 4*conv[46] + 2*conv[52] + 64*conv[53]
    conv_mod[25] = add(add(sub(mul(m31(2), conv_tmp[18]), mul(m31(4), conv_tmp[46])), mul(m31(2), conv_tmp[52])), mul(m31(64), conv_tmp[53]));

    // Limb 26: 2*conv[19] - 4*conv[47] + 2*conv[53] + 64*conv[54]
    conv_mod[26] = add(add(sub(mul(m31(2), conv_tmp[19]), mul(m31(4), conv_tmp[47])), mul(m31(2), conv_tmp[53])), mul(m31(64), conv_tmp[54]));

    // Limb 27: 2*conv[20] - 4*conv[48] + 2*conv[54]
    conv_mod[27] = add(sub(mul(m31(2), conv_tmp[20]), mul(m31(4), conv_tmp[48])), mul(m31(2), conv_tmp[54]));

    // Step 4: Add range check for k (verify k is in valid range)
    // k should be in range [-262144, 262144] represented as k + 262144 in [0, 524288]
    {
        m31 values[1] = {add(k, m31(262144))};
        RelationEntry<1> entry(
            mul_eval->range_check_19_h_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Step 5: Verify carry propagation constraints
    // carry_i * 512 = conv_mod[i] - k (for i=0) or conv_mod[i] + carry_{i-1} (for i>0)
    // Each carry must be range-checked to [-131072, 131072]

    m31 M31_512_f = m31(512);
    m31 M31_131072_f = m31(131072);
    m31 M31_136_f = m31(136);
    m31 M31_256_f = m31(256);

    // Carry 0: carry_0 * 512 = conv_mod[0] - k
    eval->add_constraint(sub(mul(carries[0], M31_512_f), sub(conv_mod[0], k)));
    {
        m31 values[1] = {add(carries[0], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 1: carry_1 * 512 = conv_mod[1] + carry_0
    eval->add_constraint(sub(mul(carries[1], M31_512_f), add(conv_mod[1], carries[0])));
    {
        m31 values[1] = {add(carries[1], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_b_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 2
    eval->add_constraint(sub(mul(carries[2], M31_512_f), add(conv_mod[2], carries[1])));
    {
        m31 values[1] = {add(carries[2], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_c_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 3
    eval->add_constraint(sub(mul(carries[3], M31_512_f), add(conv_mod[3], carries[2])));
    {
        m31 values[1] = {add(carries[3], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_d_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 4
    eval->add_constraint(sub(mul(carries[4], M31_512_f), add(conv_mod[4], carries[3])));
    {
        m31 values[1] = {add(carries[4], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_e_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 5
    eval->add_constraint(sub(mul(carries[5], M31_512_f), add(conv_mod[5], carries[4])));
    {
        m31 values[1] = {add(carries[5], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_f_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 6
    eval->add_constraint(sub(mul(carries[6], M31_512_f), add(conv_mod[6], carries[5])));
    {
        m31 values[1] = {add(carries[6], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_g_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 7
    eval->add_constraint(sub(mul(carries[7], M31_512_f), add(conv_mod[7], carries[6])));
    {
        m31 values[1] = {add(carries[7], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_h_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 8
    eval->add_constraint(sub(mul(carries[8], M31_512_f), add(conv_mod[8], carries[7])));
    {
        m31 values[1] = {add(carries[8], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 9
    eval->add_constraint(sub(mul(carries[9], M31_512_f), add(conv_mod[9], carries[8])));
    {
        m31 values[1] = {add(carries[9], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_b_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 10
    eval->add_constraint(sub(mul(carries[10], M31_512_f), add(conv_mod[10], carries[9])));
    {
        m31 values[1] = {add(carries[10], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_c_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 11
    eval->add_constraint(sub(mul(carries[11], M31_512_f), add(conv_mod[11], carries[10])));
    {
        m31 values[1] = {add(carries[11], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_d_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 12
    eval->add_constraint(sub(mul(carries[12], M31_512_f), add(conv_mod[12], carries[11])));
    {
        m31 values[1] = {add(carries[12], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_e_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 13
    eval->add_constraint(sub(mul(carries[13], M31_512_f), add(conv_mod[13], carries[12])));
    {
        m31 values[1] = {add(carries[13], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_f_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 14
    eval->add_constraint(sub(mul(carries[14], M31_512_f), add(conv_mod[14], carries[13])));
    {
        m31 values[1] = {add(carries[14], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_g_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 15
    eval->add_constraint(sub(mul(carries[15], M31_512_f), add(conv_mod[15], carries[14])));
    {
        m31 values[1] = {add(carries[15], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_h_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 16
    eval->add_constraint(sub(mul(carries[16], M31_512_f), add(conv_mod[16], carries[15])));
    {
        m31 values[1] = {add(carries[16], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 17
    eval->add_constraint(sub(mul(carries[17], M31_512_f), add(conv_mod[17], carries[16])));
    {
        m31 values[1] = {add(carries[17], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_b_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 18
    eval->add_constraint(sub(mul(carries[18], M31_512_f), add(conv_mod[18], carries[17])));
    {
        m31 values[1] = {add(carries[18], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_c_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 19
    eval->add_constraint(sub(mul(carries[19], M31_512_f), add(conv_mod[19], carries[18])));
    {
        m31 values[1] = {add(carries[19], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_d_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 20
    eval->add_constraint(sub(mul(carries[20], M31_512_f), add(conv_mod[20], carries[19])));
    {
        m31 values[1] = {add(carries[20], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_e_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 21 (special: subtract 136*k)
    eval->add_constraint(sub(mul(carries[21], M31_512_f), add(sub(conv_mod[21], mul(M31_136_f, k)), carries[20])));
    {
        m31 values[1] = {add(carries[21], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_f_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 22
    eval->add_constraint(sub(mul(carries[22], M31_512_f), add(conv_mod[22], carries[21])));
    {
        m31 values[1] = {add(carries[22], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_g_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 23
    eval->add_constraint(sub(mul(carries[23], M31_512_f), add(conv_mod[23], carries[22])));
    {
        m31 values[1] = {add(carries[23], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_h_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 24
    eval->add_constraint(sub(mul(carries[24], M31_512_f), add(conv_mod[24], carries[23])));
    {
        m31 values[1] = {add(carries[24], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 25
    eval->add_constraint(sub(mul(carries[25], M31_512_f), add(conv_mod[25], carries[24])));
    {
        m31 values[1] = {add(carries[25], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_b_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Carry 26
    eval->add_constraint(sub(mul(carries[26], M31_512_f), add(conv_mod[26], carries[25])));
    {
        m31 values[1] = {add(carries[26], M31_131072_f)};
        RelationEntry<1> entry(
            mul_eval->range_check_19_c_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
        eval->add_to_relation<1>(entry);
    }

    // Final constraint: conv_mod[27] - 256*k + carry_26 = 0
    eval->add_constraint(add(sub(conv_mod[27], mul(M31_256_f, k)), carries[26]));
}

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__global__ void evaluate_mul_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MulOpcode_Eval *mul_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT cuda_evaluator(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Read all 130 trace columns (trace1)
    // Columns 0-2: Input state (PC, AP, FP)
    m31 input_pc = cuda_evaluator.next_trace_mask();
    m31 input_ap = cuda_evaluator.next_trace_mask();
    m31 input_fp = cuda_evaluator.next_trace_mask();

    // Columns 3-10: Decode flags and offsets
    m31 offset0 = cuda_evaluator.next_trace_mask();
    m31 offset1 = cuda_evaluator.next_trace_mask();
    m31 offset2 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp = cuda_evaluator.next_trace_mask();
    m31 op1_imm = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1 = cuda_evaluator.next_trace_mask();

    // Columns 11-13: Memory base addresses
    m31 mem_dst_base = cuda_evaluator.next_trace_mask();
    m31 mem0_base = cuda_evaluator.next_trace_mask();
    m31 mem1_base = cuda_evaluator.next_trace_mask();

    // Columns 14-42: dst (28 limbs + id)
    m31 dst_id = cuda_evaluator.next_trace_mask();
    m31 dst_limbs[28];
    for (int i = 0; i < 28; i++) {
        dst_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 43-71: op0 (28 limbs + id)
    m31 op0_id = cuda_evaluator.next_trace_mask();
    m31 op0_limbs[28];
    for (int i = 0; i < 28; i++) {
        op0_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 72-100: op1 (28 limbs + id)
    m31 op1_id = cuda_evaluator.next_trace_mask();
    m31 op1_limbs[28];
    for (int i = 0; i < 28; i++) {
        op1_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Column 101: k (modular reduction parameter)
    m31 k = cuda_evaluator.next_trace_mask();

    // Columns 102-128: carries (27 carry values)
    m31 carries[27];
    for (int i = 0; i < 27; i++) {
        carries[i] = cuda_evaluator.next_trace_mask();
    }

    // Column 129: enabler
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Constraint 0: enabler is boolean
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);

    // Call DecodeInstruction4B8Cf subroutine
    m31 decode_outputs[19];
    evaluate_decode_instruction_4b8cf(
        input_pc,
        offset0, offset1, offset2,
        dst_base_fp, op0_base_fp,
        op1_imm, op1_base_fp,
        ap_update_add_1,
        decode_outputs,
        mul_eval->verify_instruction_lookup_elements,
        &cuda_evaluator
    );

    m31 decode_offset0 = decode_outputs[0];
    m31 decode_offset1 = decode_outputs[1];
    m31 decode_offset2 = decode_outputs[2];
    // Note: CUDA decode function stores op1_base_ap at index 7
    // (1 - op1_imm) - op1_base_fp = op1_base_ap
    m31 decode_op1_base_ap = decode_outputs[7];

    // Constraint 1: if imm then offset2 is 1
    cuda_evaluator.add_constraint(mul(op1_imm, sub(M31_1, decode_offset2)));

    // Constraint 2: mem_dst_base
    cuda_evaluator.add_constraint(
        sub(mem_dst_base, add(mul(dst_base_fp, input_fp), mul(sub(M31_1, dst_base_fp), input_ap)))
    );

    // Constraint 3: mem0_base
    cuda_evaluator.add_constraint(
        sub(mem0_base, add(mul(op0_base_fp, input_fp), mul(sub(M31_1, op0_base_fp), input_ap)))
    );

    // Constraint 4: mem1_base
    cuda_evaluator.add_constraint(
        sub(mem1_base, add(add(mul(op1_imm, input_pc), mul(op1_base_fp, input_fp)), mul(decode_op1_base_ap, input_ap)))
    );

    // Call ReadPositiveNumBits252 for dst
    m31 dst_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem_dst_base, decode_offset0),
        dst_id,
        dst_limbs[0], dst_limbs[1], dst_limbs[2], dst_limbs[3],
        dst_limbs[4], dst_limbs[5], dst_limbs[6], dst_limbs[7],
        dst_limbs[8], dst_limbs[9], dst_limbs[10], dst_limbs[11],
        dst_limbs[12], dst_limbs[13], dst_limbs[14], dst_limbs[15],
        dst_limbs[16], dst_limbs[17], dst_limbs[18], dst_limbs[19],
        dst_limbs[20], dst_limbs[21], dst_limbs[22], dst_limbs[23],
        dst_limbs[24], dst_limbs[25], dst_limbs[26], dst_limbs[27],
        dst_output,
        mul_eval->memory_address_to_id_lookup_elements,
        mul_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator
    );

    // Call ReadPositiveNumBits252 for op0
    m31 op0_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem0_base, decode_offset1),
        op0_id,
        op0_limbs[0], op0_limbs[1], op0_limbs[2], op0_limbs[3],
        op0_limbs[4], op0_limbs[5], op0_limbs[6], op0_limbs[7],
        op0_limbs[8], op0_limbs[9], op0_limbs[10], op0_limbs[11],
        op0_limbs[12], op0_limbs[13], op0_limbs[14], op0_limbs[15],
        op0_limbs[16], op0_limbs[17], op0_limbs[18], op0_limbs[19],
        op0_limbs[20], op0_limbs[21], op0_limbs[22], op0_limbs[23],
        op0_limbs[24], op0_limbs[25], op0_limbs[26], op0_limbs[27],
        op0_output,
        mul_eval->memory_address_to_id_lookup_elements,
        mul_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator
    );

    // Call ReadPositiveNumBits252 for op1
    m31 op1_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem1_base, decode_offset2),
        op1_id,
        op1_limbs[0], op1_limbs[1], op1_limbs[2], op1_limbs[3],
        op1_limbs[4], op1_limbs[5], op1_limbs[6], op1_limbs[7],
        op1_limbs[8], op1_limbs[9], op1_limbs[10], op1_limbs[11],
        op1_limbs[12], op1_limbs[13], op1_limbs[14], op1_limbs[15],
        op1_limbs[16], op1_limbs[17], op1_limbs[18], op1_limbs[19],
        op1_limbs[20], op1_limbs[21], op1_limbs[22], op1_limbs[23],
        op1_limbs[24], op1_limbs[25], op1_limbs[26], op1_limbs[27],
        op1_output,
        mul_eval->memory_address_to_id_lookup_elements,
        mul_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator
    );

    // Call VerifyMul252: verify op0 * op1 = dst (mod Cairo prime)
    VerifyMul252<EvaluatorT>(
        &cuda_evaluator,
        op0_limbs,
        op1_limbs,
        dst_limbs,
        k,
        carries,
        mul_eval
    );

    // Add opcodes relation entries (state transition)
    // Forward entry: (input_pc, input_ap, input_fp) with multiplicity +enabler
    {
        m31 values[3] = {input_pc, input_ap, input_fp};
        RelationEntry<3> entry(
            mul_eval->opcode_lookup_elements,
            qm31{enabler, M31_0},
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    // Backward entry: (next_pc, next_ap, input_fp) with multiplicity -enabler
    m31 next_pc = add(add(input_pc, M31_1), op1_imm);
    m31 next_ap = add(input_ap, ap_update_add_1);
    {
        m31 values[3] = {next_pc, next_ap, input_fp};
        RelationEntry<3> entry(
            mul_eval->opcode_lookup_elements,
            sub(qm31{{M31_0, M31_0}, {M31_0, M31_0}}, qm31{enabler, M31_0}),
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    // Store constraint index
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
    // numerators[row] = cuda_evaluator.numerator;
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
extern "C"
void evaluate_mul_opcode(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
) {
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    MulOpcode_Eval *device_mul_eval = cuda_malloc<MulOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<MulOpcode_Eval>((MulOpcode_Eval*)eval, device_mul_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_mul_opcode");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_mul_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_mul_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    } else {
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim, 0, stream>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        g_should_accumulate_host  // Read from global variable set by dispatcher
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_mul_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_mul_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
