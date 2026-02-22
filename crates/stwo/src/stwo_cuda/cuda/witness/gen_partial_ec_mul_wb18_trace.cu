/**
 * CUDA trace generation for partial_ec_mul_window_bits_18 (297-col "now" architecture).
 *
 * Implements AIR-compatible trace generation with:
 *   - 297 base trace columns
 *   - 65 logup interaction columns (stub)
 *   - Sub-component feeds to pedersen_points_table, rc_9_9, and rc_20
 *
 * Column layout:
 *   0-71:    Input (72 cols)
 *   72-99:   Table point x (28 limbs)
 *   100-127: Table point y (28 limbs)
 *   128-155: Slope (28 limbs)
 *   156:     VerifyMul #1 k
 *   157-183: VerifyMul #1 carries (27)
 *   184-211: Result x (28 limbs)
 *   212:     VerifyMul #2 k
 *   213-239: VerifyMul #2 carries (27)
 *   240-267: Result y (28 limbs)
 *   268:     VerifyMul #3 k
 *   269-295: VerifyMul #3 carries (27)
 *   296:     Enabler
 *
 * EC point addition:
 *   slope = (y2 - y1) / (x2 - x1)
 *   result_x = slope^2 - x1 - x2
 *   result_y = slope * (x1 - result_x) - y1
 * where (x1,y1) = accumulator (input cols 16-71), (x2,y2) = table point (cols 72-127)
 */

#include "gen_partial_ec_mul_wb18_trace.cuh"
#include "../fields.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include "../utils.cuh"

// Pedersen table — defined in pedersen_table_init.cu
#define PEDERSEN_TABLE_N_COLUMNS 56
extern __device__ m31* g_pedersen_table_columns[PEDERSEN_TABLE_N_COLUMNS];

// Block size for kernel launch
#define WB18_BLOCK_SIZE 256

// ============================================================================
// Felt252 type and field operations
// ============================================================================

typedef ff_storage<8> Felt252Field;

static __device__ __forceinline__ Felt252Field wb18_felt_add(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

static __device__ __forceinline__ Felt252Field wb18_felt_sub(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

static __device__ __forceinline__ Felt252Field wb18_felt_to_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

static __device__ __forceinline__ Felt252Field wb18_felt_from_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

static __device__ __forceinline__ Felt252Field wb18_felt_mul(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

static __device__ __forceinline__ Felt252Field wb18_felt_inverse(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::inverse(a);
}

// ============================================================================
// Limb conversion utilities
// ============================================================================

// Convert 28 x 9-bit M31 limbs to Felt252Field (standard form)
static __device__ Felt252Field wb18_limbs28_to_felt252(const m31* limbs) {
    uint64_t accum = 0;
    int bit_pos = 0;
    Felt252Field result = {0};
    int out_idx = 0;

    for (int i = 0; i < 28 && out_idx < 8; i++) {
        accum |= ((uint64_t)limbs[i]) << bit_pos;
        bit_pos += 9;

        while (bit_pos >= 32 && out_idx < 8) {
            result.limbs[out_idx++] = (uint32_t)(accum & 0xFFFFFFFF);
            accum >>= 32;
            bit_pos -= 32;
        }
    }
    if (out_idx < 8) {
        result.limbs[out_idx] = (uint32_t)(accum & 0xFFFFFFFF);
    }

    return result;
}

// Convert Felt252Field (standard form) to 28 x 9-bit limbs
static __device__ void wb18_felt252_to_limbs28(const Felt252Field& felt, m31* limbs) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    for (int i = 0; i < 7; i++) {
        limbs[i] = (uint32_t)((val0 >> (i * 9)) & 0x1FF);
    }
    uint64_t cross01 = (val0 >> 63) | (val1 << 1);
    limbs[7] = (uint32_t)(cross01 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[8 + i] = (uint32_t)((val1 >> (8 + i * 9)) & 0x1FF);
    }
    uint64_t cross12 = (val1 >> 62) | (val2 << 2);
    limbs[14] = (uint32_t)(cross12 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[15 + i] = (uint32_t)((val2 >> (7 + i * 9)) & 0x1FF);
    }
    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    limbs[21] = (uint32_t)(cross23 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[22 + i] = (uint32_t)((val3 >> (6 + i * 9)) & 0x1FF);
    }
}

// ============================================================================
// Schoolbook multiplication: 28 x 28 limbs -> 55 coefficients
// Both factors are int64 arrays (can be signed for limb differences)
// ============================================================================

static __device__ void wb18_schoolbook_mul_28x28(
    const int64_t* a_limbs,
    const int64_t* b_limbs,
    int64_t* product  // 55 coefficients output
) {
    for (int i = 0; i < 55; i++) product[i] = 0;

    for (int i = 0; i < 28; i++) {
        int64_t ai = a_limbs[i];
        for (int j = 0; j < 28; j++) {
            product[i + j] += ai * b_limbs[j];
        }
    }
}

// ============================================================================
// Modular reduction: conv (55 coefficients) -> conv_mod (28 values)
// For Starknet prime p = 2^252 + 17*2^192 + 1
// ============================================================================

static __device__ void wb18_compute_conv_mod(
    const int64_t* conv,
    int64_t* conv_mod
) {
    conv_mod[0]  = 32 * conv[0] - 4 * conv[21] + 8 * conv[49];
    conv_mod[1]  = conv[0] + 32 * conv[1] - 4 * conv[22] + 8 * conv[50];
    conv_mod[2]  = conv[1] + 32 * conv[2] - 4 * conv[23] + 8 * conv[51];
    conv_mod[3]  = conv[2] + 32 * conv[3] - 4 * conv[24] + 8 * conv[52];
    conv_mod[4]  = conv[3] + 32 * conv[4] - 4 * conv[25] + 8 * conv[53];
    conv_mod[5]  = conv[4] + 32 * conv[5] - 4 * conv[26] + 8 * conv[54];
    conv_mod[6]  = conv[5] + 32 * conv[6] - 4 * conv[27];
    conv_mod[7]  = 2 * conv[0] + conv[6] + 32 * conv[7] - 4 * conv[28];
    conv_mod[8]  = 2 * conv[1] + conv[7] + 32 * conv[8] - 4 * conv[29];
    conv_mod[9]  = 2 * conv[2] + conv[8] + 32 * conv[9] - 4 * conv[30];
    conv_mod[10] = 2 * conv[3] + conv[9] + 32 * conv[10] - 4 * conv[31];
    conv_mod[11] = 2 * conv[4] + conv[10] + 32 * conv[11] - 4 * conv[32];
    conv_mod[12] = 2 * conv[5] + conv[11] + 32 * conv[12] - 4 * conv[33];
    conv_mod[13] = 2 * conv[6] + conv[12] + 32 * conv[13] - 4 * conv[34];
    conv_mod[14] = 2 * conv[7] + conv[13] + 32 * conv[14] - 4 * conv[35];
    conv_mod[15] = 2 * conv[8] + conv[14] + 32 * conv[15] - 4 * conv[36];
    conv_mod[16] = 2 * conv[9] + conv[15] + 32 * conv[16] - 4 * conv[37];
    conv_mod[17] = 2 * conv[10] + conv[16] + 32 * conv[17] - 4 * conv[38];
    conv_mod[18] = 2 * conv[11] + conv[17] + 32 * conv[18] - 4 * conv[39];
    conv_mod[19] = 2 * conv[12] + conv[18] + 32 * conv[19] - 4 * conv[40];
    conv_mod[20] = 2 * conv[13] + conv[19] + 32 * conv[20] - 4 * conv[41];
    conv_mod[21] = 2 * conv[14] + conv[20] - 4 * conv[42] + 64 * conv[49];
    conv_mod[22] = 2 * conv[15] - 4 * conv[43] + 2 * conv[49] + 64 * conv[50];
    conv_mod[23] = 2 * conv[16] - 4 * conv[44] + 2 * conv[50] + 64 * conv[51];
    conv_mod[24] = 2 * conv[17] - 4 * conv[45] + 2 * conv[51] + 64 * conv[52];
    conv_mod[25] = 2 * conv[18] - 4 * conv[46] + 2 * conv[52] + 64 * conv[53];
    conv_mod[26] = 2 * conv[19] - 4 * conv[47] + 2 * conv[53] + 64 * conv[54];
    conv_mod[27] = 2 * conv[20] - 4 * conv[48] + 2 * conv[54];
}

// ============================================================================
// Extract k value from conv_mod using biased arithmetic
// ============================================================================

static __device__ __forceinline__ int64_t wb18_compute_k(const int64_t* conv_mod) {
    uint32_t k_mod_tmp = (
        (uint32_t)(conv_mod[0] + 134217728) +
        (((uint32_t)(conv_mod[1] + 134217728) & 511) << 9) +
        131072
    ) & 262143;

    int64_t k_val = (int64_t)(k_mod_tmp & 0xFFFF) +
                    (int64_t)((int32_t)((k_mod_tmp >> 16) & 0x3) - 2) * 65536;
    return k_val;
}

// ============================================================================
// Compute carry chain from conv_mod and k
// ============================================================================

static __device__ void wb18_compute_carries(
    const int64_t* conv_mod,
    int64_t k_val,
    int64_t* carry  // 27 carry values
) {
    carry[0] = (conv_mod[0] - k_val) / 512;
    for (int i = 1; i < 21; i++) {
        carry[i] = (conv_mod[i] + carry[i-1]) / 512;
    }
    // Special case at carry[21]: includes -136*k term
    carry[21] = (conv_mod[21] - 136 * k_val + carry[20]) / 512;
    for (int i = 22; i < 27; i++) {
        carry[i] = (conv_mod[i] + carry[i-1]) / 512;
    }
}

// ============================================================================
// Convert int64 to M31 (proper modular reduction)
// ============================================================================

static __device__ __forceinline__ m31 wb18_int64_to_m31(int64_t val) {
    const int64_t P = 2147483647LL;  // 2^31 - 1
    int64_t result = val % P;
    if (result < 0) result += P;
    return (m31)(uint32_t)result;
}

// ============================================================================
// M31 addition (modular)
// ============================================================================

static __device__ __forceinline__ m31 wb18_m31_add(m31 a, m31 b) {
    uint32_t sum = a + b;
    if (sum >= 2147483647u) sum -= 2147483647u;
    return sum;
}

// ============================================================================
// VerifyMul252: compute k and carries for proving a*b = c (mod P)
//
// All inputs are int64 arrays (can hold signed limb differences/sums).
// Computes: schoolbook(a_limbs, b_limbs) - c_limbs = k*P (in limb space)
// Outputs k as M31 and 27 carries as M31.
// ============================================================================

static __device__ void wb18_verify_mul_252(
    const int64_t* a_limbs,  // 28 limbs (first factor)
    const int64_t* b_limbs,  // 28 limbs (second factor)
    const int64_t* c_limbs,  // 28 limbs (expected: a*b = c mod P)
    m31* k_out,              // output k value
    m31* carry_out           // 27 output carry values
) {
    // Schoolbook product
    int64_t product[55];
    wb18_schoolbook_mul_28x28(a_limbs, b_limbs, product);

    // conv = product - expected
    int64_t conv[55];
    for (int i = 0; i < 28; i++) {
        conv[i] = product[i] - c_limbs[i];
    }
    for (int i = 28; i < 55; i++) {
        conv[i] = product[i];
    }

    // Modular reduction
    int64_t conv_mod[28];
    wb18_compute_conv_mod(conv, conv_mod);

    // Extract k
    int64_t k_val = wb18_compute_k(conv_mod);
    *k_out = wb18_int64_to_m31(k_val);

    // Compute carries
    int64_t carry[27];
    wb18_compute_carries(conv_mod, k_val, carry);
    for (int i = 0; i < 27; i++) {
        carry_out[i] = wb18_int64_to_m31(carry[i]);
    }
}

// ============================================================================
// Kernel arguments (passed via constant memory)
// ============================================================================

struct PemWb18Args {
    m31** traces;              // [297] device ptrs
    m31** inputs;              // [72] device ptrs
    m31** sub_ppt;             // [1] device ptr
    m31** sub_rc_9_9[8];       // 8 variant arrays
    m31** sub_rc_20[8];        // 8 variant arrays
    m31** lk_pem_0;            // [73] device ptrs
    m31** lk_pem_1;            // [73] device ptrs
    m31** lk_ppt_0;            // [58] device ptrs
    m31** lk_rc_20[8];         // 8 variant arrays
    m31** lk_rc_9_9[8];        // 8 variant arrays
    uint32_t n_rows;
    uint32_t trace_size;
};

static __constant__ PemWb18Args d_wb18_args;

// ============================================================================
// Sub-component input helpers
// ============================================================================

// Write RC_9_9 sub-component inputs for one 28-limb field element.
// field_idx: 0=slope, 1=result_x, 2=result_y
// Distribution: 14 pairs round-robin across 8 variants [a..h, a..f]
// Variant counts per field element: [2,2,2,2,2,2,1,1]
static __device__ void wb18_write_rc_9_9_sub_inputs(
    const m31* limbs,
    int field_idx,
    m31** sub_rc_9_9[8],
    uint32_t row
) {
    for (int p = 0; p < 14; p++) {
        int variant, local_idx;
        if (p < 8) {
            variant = p;
            local_idx = 0;
        } else {
            variant = p - 8;
            local_idx = 1;
        }

        int entries_per_field = (variant < 6) ? 2 : 1;
        int entry = field_idx * entries_per_field + local_idx;

        sub_rc_9_9[variant][2 * entry][row] = limbs[2 * p];
        sub_rc_9_9[variant][2 * entry + 1][row] = limbs[2 * p + 1];
    }
}

// Write RC_20 sub-component inputs for one VerifyMul (k + 27 carries = 28 values).
// vm_idx: 0, 1, or 2 (which VerifyMul)
// Distribution: 28 values round-robin across 8 variants [a..h]
// Variant counts per VerifyMul: [4,4,4,4,3,3,3,3]
static __device__ void wb18_write_rc_20_sub_inputs(
    m31 k_val,
    const m31* carries,
    int vm_idx,
    m31** sub_rc_20[8],
    uint32_t row
) {
    const uint32_t BIAS = 524288u;  // 2^19

    for (int v = 0; v < 28; v++) {
        int variant = v % 8;
        int local_idx = v / 8;

        int entries_per_vm = (variant < 4) ? 4 : 3;
        int entry = vm_idx * entries_per_vm + local_idx;

        m31 val = (v == 0) ? k_val : carries[v - 1];
        sub_rc_20[variant][entry][row] = wb18_m31_add(val, BIAS);
    }
}

// ============================================================================
// Lookup data helpers
// ============================================================================

// Relation ID constants (from SIMD reference)
#define WB18_PEM_RELATION_ID     1621226978u
#define WB18_PPT_RELATION_ID     1444721856u

// Per-variant relation IDs for RC_20 [a..h]
static __device__ const uint32_t WB18_RC_20_RELATION_IDS[8] = {
    1410849886u,  // a
    514232941u,   // b
    531010560u,   // c
    480677703u,   // d
    497455322u,   // e
    447122465u,   // f
    463900084u,   // g
    682009131u    // h
};

// Per-variant relation IDs for RC_9_9 [a..h]
static __device__ const uint32_t WB18_RC_9_9_RELATION_IDS[8] = {
    517791011u, 1897792095u, 1881014476u, 1864236857u,
    1847459238u, 1830681619u, 1813904000u, 2065568285u
};

// Write RC_9_9 lookup data for one 28-limb field element.
static __device__ void wb18_write_rc_9_9_lookup(
    const m31* limbs,
    int field_idx,
    m31** lk_rc_9_9[8],
    uint32_t row
) {
    for (int p = 0; p < 14; p++) {
        int variant, local_idx;
        if (p < 8) {
            variant = p;
            local_idx = 0;
        } else {
            variant = p - 8;
            local_idx = 1;
        }

        int entries_per_field = (variant < 6) ? 2 : 1;
        int entry = field_idx * entries_per_field + local_idx;

        lk_rc_9_9[variant][3 * entry][row] = WB18_RC_9_9_RELATION_IDS[variant];
        lk_rc_9_9[variant][3 * entry + 1][row] = limbs[2 * p];
        lk_rc_9_9[variant][3 * entry + 2][row] = limbs[2 * p + 1];
    }
}

// Write RC_20 lookup data for one VerifyMul.
static __device__ void wb18_write_rc_20_lookup(
    m31 k_val,
    const m31* carries,
    int vm_idx,
    m31** lk_rc_20[8],
    uint32_t row
) {
    const uint32_t BIAS = 524288u;

    for (int v = 0; v < 28; v++) {
        int variant = v % 8;
        int local_idx = v / 8;

        int entries_per_vm = (variant < 4) ? 4 : 3;
        int entry = vm_idx * entries_per_vm + local_idx;

        m31 val = (v == 0) ? k_val : carries[v - 1];

        lk_rc_20[variant][2 * entry][row] = WB18_RC_20_RELATION_IDS[variant];
        lk_rc_20[variant][2 * entry + 1][row] = wb18_m31_add(val, BIAS);
    }
}

// ============================================================================
// Main trace generation kernel
// ============================================================================

__global__ void __launch_bounds__(WB18_BLOCK_SIZE, 2)
wb18_trace_kernel() {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_wb18_args.trace_size) return;

    m31** traces = d_wb18_args.traces;
    m31** inputs = d_wb18_args.inputs;
    uint32_t n_rows = d_wb18_args.n_rows;

    // Padding rows use the input data directly (already padded by Rust with
    // first-packed-row cycling, matching the SIMD path).
    uint32_t src_row = row;

    // ====================================================================
    // 1. Read and write 72 input columns (cols 0-71)
    // ====================================================================
    m31 input_vals[72];
    for (int i = 0; i < 72; i++) {
        input_vals[i] = inputs[i][src_row];
        traces[i][row] = input_vals[i];
    }

    // ====================================================================
    // 2. Compute table index and lookup point from pedersen table
    // ====================================================================
    uint32_t table_idx = 262144u * (uint32_t)input_vals[1] + (uint32_t)input_vals[2];

    m31 table_x_limbs[28], table_y_limbs[28];
    for (int i = 0; i < 28; i++) {
        table_x_limbs[i] = g_pedersen_table_columns[i][table_idx];
    }
    for (int i = 0; i < 28; i++) {
        table_y_limbs[i] = g_pedersen_table_columns[28 + i][table_idx];
    }

    // Write table point to trace (cols 72-127)
    for (int i = 0; i < 28; i++) traces[72 + i][row] = table_x_limbs[i];
    for (int i = 0; i < 28; i++) traces[100 + i][row] = table_y_limbs[i];

    // ====================================================================
    // 3. Store raw limb arrays and convert to Felt252 for field ops
    // ====================================================================
    // acc_x limbs = input cols 16-43, acc_y limbs = input cols 44-71
    m31* acc_x_limbs = &input_vals[16];   // 28 limbs
    m31* acc_y_limbs = &input_vals[44];   // 28 limbs

    Felt252Field acc_x = wb18_limbs28_to_felt252(acc_x_limbs);
    Felt252Field acc_y = wb18_limbs28_to_felt252(acc_y_limbs);
    Felt252Field table_x = wb18_limbs28_to_felt252(table_x_limbs);
    Felt252Field table_y = wb18_limbs28_to_felt252(table_y_limbs);

    // ====================================================================
    // 4. Compute slope = (y2 - y1) / (x2 - x1)
    // ====================================================================
    Felt252Field dy = wb18_felt_sub(table_y, acc_y);
    Felt252Field dx = wb18_felt_sub(table_x, acc_x);

    // Compute slope via Montgomery field arithmetic
    Felt252Field num_mont = wb18_felt_to_mont(dy);
    Felt252Field denom_mont = wb18_felt_to_mont(dx);
    Felt252Field inv_denom_mont = wb18_felt_inverse(denom_mont);
    Felt252Field slope_mont = wb18_felt_mul(num_mont, inv_denom_mont);
    Felt252Field slope = wb18_felt_from_mont(slope_mont);

    // Decompose slope to 9-bit canonical limbs
    m31 slope_limbs[28];
    wb18_felt252_to_limbs28(slope, slope_limbs);

    // Write slope to trace (cols 128-155)
    for (int i = 0; i < 28; i++) traces[128 + i][row] = slope_limbs[i];

    // ====================================================================
    // 5. VerifyMul #1: slope * (table_x - acc_x) = (table_y - acc_y) (mod P)
    //
    // The AIR evaluates: schoolbook(slope, table_x_limbs - acc_x_limbs)
    //                   - (table_y_limbs - acc_y_limbs)
    // Using raw limb-by-limb differences (not canonical Felt252 decomposition).
    // ====================================================================
    int64_t slope_i64[28], dx_limbs[28], dy_limbs[28];
    for (int i = 0; i < 28; i++) {
        slope_i64[i] = (int64_t)slope_limbs[i];
        dx_limbs[i] = (int64_t)table_x_limbs[i] - (int64_t)acc_x_limbs[i];
        dy_limbs[i] = (int64_t)table_y_limbs[i] - (int64_t)acc_y_limbs[i];
    }

    m31 k1;
    m31 carries1[27];
    wb18_verify_mul_252(slope_i64, dx_limbs, dy_limbs, &k1, carries1);

    // Write VerifyMul #1: k (col 156), carries (cols 157-183)
    traces[156][row] = k1;
    for (int i = 0; i < 27; i++) traces[157 + i][row] = carries1[i];

    // ====================================================================
    // 6. Compute result_x = slope^2 - acc_x - table_x
    // ====================================================================
    Felt252Field slope_sq = wb18_felt_from_mont(
        wb18_felt_mul(wb18_felt_to_mont(slope), wb18_felt_to_mont(slope)));
    Felt252Field result_x_felt = wb18_felt_sub(wb18_felt_sub(slope_sq, acc_x), table_x);
    m31 result_x_limbs[28];
    wb18_felt252_to_limbs28(result_x_felt, result_x_limbs);

    // Write result_x to trace (cols 184-211)
    for (int i = 0; i < 28; i++) traces[184 + i][row] = result_x_limbs[i];

    // ====================================================================
    // 7. VerifyMul #2: slope * slope = acc_x + table_x + result_x (mod P)
    //
    // The AIR evaluates: schoolbook(slope, slope)
    //                   - (acc_x_limbs + table_x_limbs + result_x_limbs)
    // Using raw limb-by-limb sum (not canonical Felt252 decomposition).
    // ====================================================================
    int64_t vm2_expected[28];
    for (int i = 0; i < 28; i++) {
        vm2_expected[i] = (int64_t)acc_x_limbs[i] + (int64_t)table_x_limbs[i]
                        + (int64_t)result_x_limbs[i];
    }

    m31 k2;
    m31 carries2[27];
    wb18_verify_mul_252(slope_i64, slope_i64, vm2_expected, &k2, carries2);

    // Write VerifyMul #2: k (col 212), carries (cols 213-239)
    traces[212][row] = k2;
    for (int i = 0; i < 27; i++) traces[213 + i][row] = carries2[i];

    // ====================================================================
    // 8. Compute result_y = slope * (acc_x - result_x) - acc_y
    // ====================================================================
    Felt252Field slope_times_diff = wb18_felt_from_mont(
        wb18_felt_mul(wb18_felt_to_mont(slope),
                      wb18_felt_to_mont(wb18_felt_sub(acc_x, result_x_felt))));
    Felt252Field result_y_felt = wb18_felt_sub(slope_times_diff, acc_y);
    m31 result_y_limbs[28];
    wb18_felt252_to_limbs28(result_y_felt, result_y_limbs);

    // Write result_y to trace (cols 240-267)
    for (int i = 0; i < 28; i++) traces[240 + i][row] = result_y_limbs[i];

    // ====================================================================
    // 9. VerifyMul #3: slope * (acc_x - result_x) = acc_y + result_y (mod P)
    //
    // The AIR evaluates: schoolbook(slope, acc_x_limbs - result_x_limbs)
    //                   - (acc_y_limbs + result_y_limbs)
    // Using raw limb-by-limb operations.
    // ====================================================================
    int64_t vm3_b[28], vm3_expected[28];
    for (int i = 0; i < 28; i++) {
        vm3_b[i] = (int64_t)acc_x_limbs[i] - (int64_t)result_x_limbs[i];
        vm3_expected[i] = (int64_t)acc_y_limbs[i] + (int64_t)result_y_limbs[i];
    }

    m31 k3;
    m31 carries3[27];
    wb18_verify_mul_252(slope_i64, vm3_b, vm3_expected, &k3, carries3);

    // Write VerifyMul #3: k (col 268), carries (cols 269-295)
    traces[268][row] = k3;
    for (int i = 0; i < 27; i++) traces[269 + i][row] = carries3[i];

    // ====================================================================
    // 9. Enabler (col 296)
    // ====================================================================
    traces[296][row] = (row < n_rows) ? 1u : 0u;

    // ====================================================================
    // 10. Sub-component inputs
    // ====================================================================

    // PPT: table index
    d_wb18_args.sub_ppt[0][row] = table_idx;

    // RC_9_9: distribute slope/result_x/result_y limb pairs across 8 variants
    wb18_write_rc_9_9_sub_inputs(slope_limbs, 0, d_wb18_args.sub_rc_9_9, row);
    wb18_write_rc_9_9_sub_inputs(result_x_limbs, 1, d_wb18_args.sub_rc_9_9, row);
    wb18_write_rc_9_9_sub_inputs(result_y_limbs, 2, d_wb18_args.sub_rc_9_9, row);

    // RC_20: distribute k+carries from 3 VerifyMuls across 8 variants
    wb18_write_rc_20_sub_inputs(k1, carries1, 0, d_wb18_args.sub_rc_20, row);
    wb18_write_rc_20_sub_inputs(k2, carries2, 1, d_wb18_args.sub_rc_20, row);
    wb18_write_rc_20_sub_inputs(k3, carries3, 2, d_wb18_args.sub_rc_20, row);

    // ====================================================================
    // 11. Lookup data (for interaction trace)
    // ====================================================================

    // partial_ec_mul_0: [relation_id, input_0..71]
    d_wb18_args.lk_pem_0[0][row] = WB18_PEM_RELATION_ID;
    for (int i = 0; i < 72; i++) {
        d_wb18_args.lk_pem_0[1 + i][row] = input_vals[i];
    }

    // partial_ec_mul_1: [relation_id, input_0, input_1+1, input_3..15, 0, rx..., ry...]
    d_wb18_args.lk_pem_1[0][row] = WB18_PEM_RELATION_ID;
    d_wb18_args.lk_pem_1[1][row] = input_vals[0];
    d_wb18_args.lk_pem_1[2][row] = wb18_m31_add(input_vals[1], 1u);
    for (int i = 3; i < 16; i++) {
        d_wb18_args.lk_pem_1[i][row] = input_vals[i];
    }
    d_wb18_args.lk_pem_1[16][row] = 0u;  // zero
    for (int i = 0; i < 28; i++) {
        d_wb18_args.lk_pem_1[17 + i][row] = result_x_limbs[i];
    }
    for (int i = 0; i < 28; i++) {
        d_wb18_args.lk_pem_1[45 + i][row] = result_y_limbs[i];
    }

    // pedersen_points_table_0: [relation_id, table_idx, output_0..55]
    d_wb18_args.lk_ppt_0[0][row] = WB18_PPT_RELATION_ID;
    d_wb18_args.lk_ppt_0[1][row] = table_idx;
    for (int i = 0; i < 28; i++) {
        d_wb18_args.lk_ppt_0[2 + i][row] = table_x_limbs[i];
    }
    for (int i = 0; i < 28; i++) {
        d_wb18_args.lk_ppt_0[30 + i][row] = table_y_limbs[i];
    }

    // RC_20 lookup data
    wb18_write_rc_20_lookup(k1, carries1, 0, d_wb18_args.lk_rc_20, row);
    wb18_write_rc_20_lookup(k2, carries2, 1, d_wb18_args.lk_rc_20, row);
    wb18_write_rc_20_lookup(k3, carries3, 2, d_wb18_args.lk_rc_20, row);

    // RC_9_9 lookup data
    wb18_write_rc_9_9_lookup(slope_limbs, 0, d_wb18_args.lk_rc_9_9, row);
    wb18_write_rc_9_9_lookup(result_x_limbs, 1, d_wb18_args.lk_rc_9_9, row);
    wb18_write_rc_9_9_lookup(result_y_limbs, 2, d_wb18_args.lk_rc_9_9, row);
}

// ============================================================================
// Exported C functions
// ============================================================================

extern "C" void gen_partial_ec_mul_wb18_trace(
    m31** traces,
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_ppt_0,
    m31** lookup_rc_20,
    m31** lookup_rc_20_b,
    m31** lookup_rc_20_c,
    m31** lookup_rc_20_d,
    m31** lookup_rc_20_e,
    m31** lookup_rc_20_f,
    m31** lookup_rc_20_g,
    m31** lookup_rc_20_h,
    m31** lookup_rc_9_9,
    m31** lookup_rc_9_9_b,
    m31** lookup_rc_9_9_c,
    m31** lookup_rc_9_9_d,
    m31** lookup_rc_9_9_e,
    m31** lookup_rc_9_9_f,
    m31** lookup_rc_9_9_g,
    m31** lookup_rc_9_9_h,
    m31** sub_inputs_ppt,
    m31** sub_inputs_rc_9_9,
    m31** sub_inputs_rc_9_9_b,
    m31** sub_inputs_rc_9_9_c,
    m31** sub_inputs_rc_9_9_d,
    m31** sub_inputs_rc_9_9_e,
    m31** sub_inputs_rc_9_9_f,
    m31** sub_inputs_rc_9_9_g,
    m31** sub_inputs_rc_9_9_h,
    m31** sub_inputs_rc_20,
    m31** sub_inputs_rc_20_b,
    m31** sub_inputs_rc_20_c,
    m31** sub_inputs_rc_20_d,
    m31** sub_inputs_rc_20_e,
    m31** sub_inputs_rc_20_f,
    m31** sub_inputs_rc_20_g,
    m31** sub_inputs_rc_20_h,
    m31** inputs,
    uint32_t n_rows,
    uint32_t log_size
) {
    uint32_t trace_size = 1u << log_size;

    // Increase stack size for the kernel.
    // The kernel uses large local arrays (schoolbook ~3KB) plus deep call stacks
    // for Felt252 field operations (inverse requires many multiplications).
    size_t prev_stack_size = 0;
    cudaDeviceGetLimit(&prev_stack_size, cudaLimitStackSize);
    size_t needed_stack = 32768;
    if (prev_stack_size < needed_stack) {
        cudaDeviceSetLimit(cudaLimitStackSize, needed_stack);
    }

    // Copy all host pointer arrays to device
    m31** d_traces = clone_to_device<m31*>(traces, 297);
    m31** d_inputs = clone_to_device<m31*>(inputs, 72);

    m31** d_sub_ppt = clone_to_device<m31*>(sub_inputs_ppt, 1);

    m31** d_sub_rc_9_9[8] = {
        clone_to_device<m31*>(sub_inputs_rc_9_9,   12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_b,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_c,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_d,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_e,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_f,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_g,  6),
        clone_to_device<m31*>(sub_inputs_rc_9_9_h,  6),
    };

    m31** d_sub_rc_20[8] = {
        clone_to_device<m31*>(sub_inputs_rc_20,   12),
        clone_to_device<m31*>(sub_inputs_rc_20_b,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_c,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_d,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_e,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_f,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_g,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_h,  9),
    };

    m31** d_lk_pem_0 = clone_to_device<m31*>(lookup_partial_ec_mul_0, 73);
    m31** d_lk_pem_1 = clone_to_device<m31*>(lookup_partial_ec_mul_1, 73);
    m31** d_lk_ppt_0 = clone_to_device<m31*>(lookup_ppt_0, 58);

    m31** d_lk_rc_20[8] = {
        clone_to_device<m31*>(lookup_rc_20,   24),
        clone_to_device<m31*>(lookup_rc_20_b,  24),
        clone_to_device<m31*>(lookup_rc_20_c,  24),
        clone_to_device<m31*>(lookup_rc_20_d,  24),
        clone_to_device<m31*>(lookup_rc_20_e,  18),
        clone_to_device<m31*>(lookup_rc_20_f,  18),
        clone_to_device<m31*>(lookup_rc_20_g,  18),
        clone_to_device<m31*>(lookup_rc_20_h,  18),
    };

    m31** d_lk_rc_9_9[8] = {
        clone_to_device<m31*>(lookup_rc_9_9,   18),
        clone_to_device<m31*>(lookup_rc_9_9_b,  18),
        clone_to_device<m31*>(lookup_rc_9_9_c,  18),
        clone_to_device<m31*>(lookup_rc_9_9_d,  18),
        clone_to_device<m31*>(lookup_rc_9_9_e,  18),
        clone_to_device<m31*>(lookup_rc_9_9_f,  18),
        clone_to_device<m31*>(lookup_rc_9_9_g,  9),
        clone_to_device<m31*>(lookup_rc_9_9_h,  9),
    };

    // Fill kernel args struct and copy to constant memory
    PemWb18Args args;
    args.traces = d_traces;
    args.inputs = d_inputs;
    args.sub_ppt = d_sub_ppt;
    for (int i = 0; i < 8; i++) {
        args.sub_rc_9_9[i] = d_sub_rc_9_9[i];
        args.sub_rc_20[i] = d_sub_rc_20[i];
    }
    args.lk_pem_0 = d_lk_pem_0;
    args.lk_pem_1 = d_lk_pem_1;
    args.lk_ppt_0 = d_lk_ppt_0;
    for (int i = 0; i < 8; i++) {
        args.lk_rc_20[i] = d_lk_rc_20[i];
        args.lk_rc_9_9[i] = d_lk_rc_9_9[i];
    }
    args.n_rows = n_rows;
    args.trace_size = trace_size;

    cudaMemcpyToSymbol(d_wb18_args, &args, sizeof(PemWb18Args));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Launch kernel
    int grid_size = (trace_size + WB18_BLOCK_SIZE - 1) / WB18_BLOCK_SIZE;
    wb18_trace_kernel<<<grid_size, WB18_BLOCK_SIZE>>>();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // Cleanup device pointer arrays
    cuda_free_memory(d_traces);
    cuda_free_memory(d_inputs);
    cuda_free_memory(d_sub_ppt);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_sub_rc_9_9[i]);
        cuda_free_memory(d_sub_rc_20[i]);
    }
    cuda_free_memory(d_lk_pem_0);
    cuda_free_memory(d_lk_pem_1);
    cuda_free_memory(d_lk_ppt_0);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_lk_rc_20[i]);
        cuda_free_memory(d_lk_rc_9_9[i]);
    }

    // Restore stack size
    if (prev_stack_size < needed_stack) {
        cudaDeviceSetLimit(cudaLimitStackSize, prev_stack_size);
    }
}

extern "C" void gen_partial_ec_mul_wb18_interaction_trace(
    void* partial_ec_mul_lookup_elements,
    void* rc_20_lookup_elements,
    void* rc_20_b_lookup_elements,
    void* rc_20_c_lookup_elements,
    void* rc_20_d_lookup_elements,
    void* rc_20_e_lookup_elements,
    void* rc_20_f_lookup_elements,
    void* rc_20_g_lookup_elements,
    void* rc_20_h_lookup_elements,
    void* rc_9_9_lookup_elements,
    void* rc_9_9_b_lookup_elements,
    void* rc_9_9_c_lookup_elements,
    void* rc_9_9_d_lookup_elements,
    void* rc_9_9_e_lookup_elements,
    void* rc_9_9_f_lookup_elements,
    void* rc_9_9_g_lookup_elements,
    void* rc_9_9_h_lookup_elements,
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_ppt_0,
    m31** lookup_rc_20,
    m31** lookup_rc_20_b,
    m31** lookup_rc_20_c,
    m31** lookup_rc_20_d,
    m31** lookup_rc_20_e,
    m31** lookup_rc_20_f,
    m31** lookup_rc_20_g,
    m31** lookup_rc_20_h,
    m31** lookup_rc_9_9,
    m31** lookup_rc_9_9_b,
    m31** lookup_rc_9_9_c,
    m31** lookup_rc_9_9_d,
    m31** lookup_rc_9_9_e,
    m31** lookup_rc_9_9_f,
    m31** lookup_rc_9_9_g,
    m31** lookup_rc_9_9_h,
    uint32_t n_rows,
    uint32_t log_size,
    m31** interaction_trace_columns,
    m31* claimed_sum
) {
    uint32_t trace_size = 1u << log_size;

    // TODO: Implement interaction trace generation using logup pattern.
    // Zero the claimed_sum for now (stub)
    uint32_t zero_sum[4] = {0, 0, 0, 0};
    cudaMemcpy(claimed_sum, zero_sum, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
}
