// CUDA kernel to generate all 30 partial_ec_mul inputs directly on GPU
// from pedersen_builtin trace data, avoiding the ~7s CPU round-trip.
//
// Each pedersen row produces 30 partial_ec_mul inputs:
//   - Chain 1 (slots 0-13):  14 rounds with P0_TABLE_OFFSET (value A low bits)
//   - Chain 2 (slot 14):      1 round  with P1_TABLE_OFFSET (value A high bits)
//   - Chain 3 (slots 15-28): 14 rounds with P2_TABLE_OFFSET (value B low bits)
//   - Chain 4 (slot 29):      1 round  with P3_TABLE_OFFSET (value B high bits)
//
// Each input has 73 columns:
//   col 0:     chain_id (row_idx * 4 + chain_offset)
//   col 1:     round
//   col 2:     table_offset
//   cols 3-16: m_shifted[14]
//   cols 17-44: acc_x (28 M31 9-bit limbs)
//   cols 45-72: acc_y (28 M31 9-bit limbs)

#include "fields.cuh"
#include "utils.cuh"
#include "pedersen_table.cuh"
#include "ec_ops.cuh"
#include <cstdint>

// Pedersen table section offsets (same as gen_pedersen_builtin_trace.cu)
#define P0_TABLE_OFFSET 0
#define P1_TABLE_OFFSET 3670016
#define P2_TABLE_OFFSET 3670032
#define P3_TABLE_OFFSET 7340048

// Shift point coordinates as 28 9-bit M31 limbs
__constant__ uint32_t PEM_SHIFT_POINT_X_M31_LIMBS[28] = {
    435, 50, 508, 83, 221, 281, 377, 383,
    212, 264, 301, 458, 130, 102, 385, 269,
    145, 276, 483, 226, 422, 253, 308, 125,
    472, 301, 227, 27
};
__constant__ uint32_t PEM_SHIFT_POINT_Y_M31_LIMBS[28] = {
    92, 321, 252, 259, 252, 413, 228, 31,
    24, 118, 301, 202, 15, 464, 334, 212,
    471, 461, 419, 354, 96, 213, 319, 191,
    251, 330, 15, 222
};

// ============================================================================
// Device function: Perform one partial_ec_mul round (same logic as pedersen
// builtin kernel's partial_ec_mul_round)
// ============================================================================
__device__ void pem_partial_ec_mul_round(
    uint32_t table_offset,
    uint32_t round,
    uint32_t window_value,
    m31* m_shifted,        // 14 elements, shifted after each round
    felt252& acc_x,
    felt252& acc_y
) {
    uint32_t table_row = table_offset + round * PEDERSEN_ROWS_PER_WINDOW + window_value;

    m31 point_x_limbs[28];
    m31 point_y_limbs[28];
    pedersen_table_lookup(table_row, point_x_limbs, point_y_limbs);

    felt252 point_x, point_y;
    felt252_from_28_limbs(point_x, point_x_limbs);
    felt252_from_28_limbs(point_y, point_y_limbs);

    AffinePointCuda table_point;
    table_point.x = point_x;
    table_point.y = point_y;

    ProjectivePointCuda acc_proj;
    acc_proj.X = felt_to_mont(acc_x);
    acc_proj.Y = felt_to_mont(acc_y);
    acc_proj.Z = ff_config_starknet::one;

    ec_add_mixed(acc_proj, table_point);

    AffinePointCuda acc_affine;
    projective_to_affine(acc_proj, acc_affine);
    acc_x = acc_affine.x;
    acc_y = acc_affine.y;

    for (int i = 0; i < 13; i++) {
        m_shifted[i] = m_shifted[i + 1];
    }
    m_shifted[13] = {0};
}

// ============================================================================
// Device function: Write 73-column input for one partial_ec_mul slot
// ============================================================================
__device__ void pem_write_input(
    m31** output,  // 73 column pointers for this slot
    uint32_t idx,
    uint32_t chain_id,
    uint32_t round,
    uint32_t table_offset,
    const m31* m_shifted,
    const felt252& acc_x,
    const felt252& acc_y
) {
    output[0][idx] = {chain_id};
    output[1][idx] = {round};
    output[2][idx] = {table_offset};

    for (int i = 0; i < 14; i++) {
        output[3 + i][idx] = m_shifted[i];
    }

    m31 x_limbs[28], y_limbs[28];
    felt252_to_28_limbs(acc_x, x_limbs);
    felt252_to_28_limbs(acc_y, y_limbs);
    for (int i = 0; i < 28; i++) {
        output[17 + i][idx] = x_limbs[i];
        output[45 + i][idx] = y_limbs[i];
    }
}

// ============================================================================
// Main kernel: Generate all 30 partial_ec_mul inputs from pedersen trace
// ============================================================================
__global__ void gen_pedersen_partial_ec_mul_inputs_kernel(
    const m31** pedersen_trace,   // pedersen_builtin trace columns
    m31*** output,                // output[30][73] column pointers
    uint32_t n_rows
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    const m31 M31_0 = {0};
    const m31 M31_512 = {512};

    // ========================================================================
    // Read value A limbs (cols 0-26), ms_limb_low_a (col 27), ms_limb_high_a (col 28)
    // ========================================================================
    m31 value_a_limbs[28];
    for (int i = 0; i < 27; i++) {
        value_a_limbs[i] = pedersen_trace[i][idx];
    }
    m31 ms_limb_low_a = pedersen_trace[27][idx];
    m31 ms_limb_high_a = pedersen_trace[28][idx];
    value_a_limbs[27] = ms_limb_low_a;  // Use ms_limb_low_a as the 28th limb

    // ========================================================================
    // Read value B limbs (cols 30-56), ms_limb_low_b (col 57), ms_limb_high_b (col 58)
    // ========================================================================
    m31 value_b_limbs[28];
    for (int i = 0; i < 27; i++) {
        value_b_limbs[i] = pedersen_trace[30 + i][idx];
    }
    m31 ms_limb_low_b = pedersen_trace[57][idx];
    m31 ms_limb_high_b = pedersen_trace[58][idx];
    value_b_limbs[27] = ms_limb_low_b;

    // ========================================================================
    // Compute m_shifted arrays (14 18-bit windows from 252-bit values)
    // ========================================================================
    m31 m_shifted_a[14];
    for (int i = 0; i < 13; i++) {
        m_shifted_a[i] = add(value_a_limbs[2*i], mul(value_a_limbs[2*i + 1], M31_512));
    }
    m_shifted_a[13] = add(value_a_limbs[26], mul(ms_limb_low_a, M31_512));

    m31 m_shifted_b[14];
    for (int i = 0; i < 13; i++) {
        m_shifted_b[i] = add(value_b_limbs[2*i], mul(value_b_limbs[2*i + 1], M31_512));
    }
    m_shifted_b[13] = add(value_b_limbs[26], mul(ms_limb_low_b, M31_512));

    // ========================================================================
    // Initialize accumulator with shift point
    // ========================================================================
    m31 shift_x_m31[28], shift_y_m31[28];
    for (int i = 0; i < 28; i++) {
        shift_x_m31[i] = {PEM_SHIFT_POINT_X_M31_LIMBS[i]};
        shift_y_m31[i] = {PEM_SHIFT_POINT_Y_M31_LIMBS[i]};
    }
    felt252 acc_x, acc_y;
    felt252_from_28_limbs(acc_x, shift_x_m31);
    felt252_from_28_limbs(acc_y, shift_y_m31);

    uint32_t row_chain_id = idx * 4;

    // ========================================================================
    // Chain 1: 14 rounds (output slots 0-13)
    // ========================================================================
    m31 chain1_m_shifted[14];
    for (int i = 0; i < 14; i++) chain1_m_shifted[i] = m_shifted_a[i];

    for (int round = 0; round < 14; round++) {
        // Write input BEFORE the round
        pem_write_input(output[round], idx,
            row_chain_id, round, P0_TABLE_OFFSET,
            chain1_m_shifted, acc_x, acc_y);

        uint32_t window_value = chain1_m_shifted[0];
        pem_partial_ec_mul_round(P0_TABLE_OFFSET, round, window_value, chain1_m_shifted, acc_x, acc_y);
    }

    // ========================================================================
    // Chain 2: 1 round (output slot 14)
    // ========================================================================
    m31 chain2_m_shifted[14];
    for (int i = 0; i < 14; i++) chain2_m_shifted[i] = M31_0;
    chain2_m_shifted[0] = ms_limb_high_a;

    pem_write_input(output[14], idx,
        row_chain_id + 1, 0, P1_TABLE_OFFSET,
        chain2_m_shifted, acc_x, acc_y);

    uint32_t window_value_a_high = ms_limb_high_a;
    pem_partial_ec_mul_round(P1_TABLE_OFFSET, 0, window_value_a_high, chain2_m_shifted, acc_x, acc_y);

    // ========================================================================
    // Chain 3: 14 rounds (output slots 15-28)
    // ========================================================================
    m31 chain3_m_shifted[14];
    for (int i = 0; i < 14; i++) chain3_m_shifted[i] = m_shifted_b[i];

    for (int round = 0; round < 14; round++) {
        pem_write_input(output[15 + round], idx,
            row_chain_id + 2, round, P2_TABLE_OFFSET,
            chain3_m_shifted, acc_x, acc_y);

        uint32_t window_value = chain3_m_shifted[0];
        pem_partial_ec_mul_round(P2_TABLE_OFFSET, round, window_value, chain3_m_shifted, acc_x, acc_y);
    }

    // ========================================================================
    // Chain 4: 1 round (output slot 29)
    // ========================================================================
    m31 chain4_m_shifted[14];
    for (int i = 0; i < 14; i++) chain4_m_shifted[i] = M31_0;
    chain4_m_shifted[0] = ms_limb_high_b;

    pem_write_input(output[29], idx,
        row_chain_id + 3, 0, P3_TABLE_OFFSET,
        chain4_m_shifted, acc_x, acc_y);
}

// ============================================================================
// C API
// ============================================================================
extern "C" void generate_pedersen_partial_ec_mul_inputs(
    const m31** pedersen_trace,     // pedersen trace column device pointers
    m31*** output,                  // output[30][73] column device pointers
    uint32_t n_trace_columns,       // number of pedersen trace columns
    uint32_t n_rows
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Clone pedersen_trace pointer array to device
    const m31** device_pedersen_trace = (const m31**)clone_to_device<const m31*>(
        const_cast<const m31**>(pedersen_trace), n_trace_columns);

    // Clone the 2D output pointer array to device
    // output is [30][73] - first clone the outer array of 30 m31** pointers,
    // then clone each inner array of 73 m31* pointers
    m31** device_inner_ptrs[30];
    for (int slot = 0; slot < 30; slot++) {
        device_inner_ptrs[slot] = clone_to_device<m31*>(
            reinterpret_cast<m31**>(output[slot]), 73);
    }
    m31*** device_output = clone_to_device<m31**>(device_inner_ptrs, 30);

    gen_pedersen_partial_ec_mul_inputs_kernel<<<num_blocks, BLOCK_SIZE>>>(
        device_pedersen_trace,
        device_output,
        n_rows
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Free device pointer arrays
    cuda_free_memory(const_cast<m31**>(device_pedersen_trace));
    for (int slot = 0; slot < 30; slot++) {
        cuda_free_memory(device_inner_ptrs[slot]);
    }
    cuda_free_memory(device_output);
}
