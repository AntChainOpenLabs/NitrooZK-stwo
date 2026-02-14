// CUDA trace generation for pedersen_builtin component
// Full GPU implementation - computes partial_ec_mul on GPU using PEDERSEN_TABLE
// 351 trace columns, ~20 interaction trace columns
//
// This implementation computes everything on GPU including EC point additions:
// - Memory address/ID lookups
// - EC point lookups from PEDERSEN_TABLE
// - EC point additions (252-bit field arithmetic)
// - Trace column generation

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"
#include "pedersen_table.cuh"
#include "ec_ops.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "../constraints/relations.cuh"
#include <cstdint>
#include <cstdio>

// Constants for memory columns
#define N_MEMORY_COLUMNS 66
#define N_PRECOMPUTED_EC_COLUMNS_DEF 285

// Pedersen table section offsets
#define P0_TABLE_OFFSET 0
#define P1_TABLE_OFFSET 3670016
#define P2_TABLE_OFFSET 3670032
#define P3_TABLE_OFFSET 7340048

// Shift point coordinates (initial accumulator) as 28 9-bit M31 limbs
// These values match the CPU implementation's hardcoded constants
__constant__ uint32_t SHIFT_POINT_X_M31_LIMBS[28] = {
    435, 50, 508, 83, 221, 281, 377, 383,
    212, 264, 301, 458, 130, 102, 385, 269,
    145, 276, 483, 226, 422, 253, 308, 125,
    472, 301, 227, 27
};
__constant__ uint32_t SHIFT_POINT_Y_M31_LIMBS[28] = {
    92, 321, 252, 259, 252, 413, 228, 31,
    24, 118, 301, 202, 15, 464, 334, 212,
    471, 461, 419, 354, 96, 213, 319, 191,
    251, 330, 15, 222
};

// ============================================================================
// Device function: Compute partial_ec_mul for one round
// ============================================================================
__device__ void partial_ec_mul_round(
    uint32_t table_offset,
    uint32_t round,
    uint32_t window_value,
    m31* m_shifted,        // 14 elements, shifted after each round
    felt252& acc_x,        // Accumulator x coordinate
    felt252& acc_y         // Accumulator y coordinate
) {
    // Compute table row index
    uint32_t table_row = table_offset + round * PEDERSEN_ROWS_PER_WINDOW + window_value;

    // Lookup point from table (28 M31 limbs for x, 28 for y)
    m31 point_x_limbs[28];
    m31 point_y_limbs[28];
    pedersen_table_lookup(table_row, point_x_limbs, point_y_limbs);

    // Convert M31 limbs to felt252
    felt252 point_x, point_y;
    felt252_from_28_limbs(point_x, point_x_limbs);
    felt252_from_28_limbs(point_y, point_y_limbs);

    // Create affine point from table
    AffinePointCuda table_point;
    table_point.x = point_x;
    table_point.y = point_y;

    // Create projective point from accumulator (convert to Montgomery form)
    ProjectivePointCuda acc_proj;
    acc_proj.X = felt_to_mont(acc_x);
    acc_proj.Y = felt_to_mont(acc_y);
    acc_proj.Z = ff_config_starknet::one;  // Z = 1 in Montgomery form

    // Perform EC point addition: acc + table_point
    ec_add_mixed(acc_proj, table_point);

    // Convert back from projective to affine using proper field inverse
    AffinePointCuda acc_affine;
    projective_to_affine(acc_proj, acc_affine);
    acc_x = acc_affine.x;
    acc_y = acc_affine.y;

    // Shift m_shifted array
    for (int i = 0; i < 13; i++) {
        m_shifted[i] = m_shifted[i + 1];
    }
    m_shifted[13] = {0};
}

// ============================================================================
// Device function: Extract 18-bit window value from 9-bit limbs
// ============================================================================
__device__ uint32_t extract_window_value(m31 limb_low, m31 limb_high) {
    // Each limb is 9 bits, combine two to get 18 bits
    return (limb_low & 0x1FF) | ((limb_high & 0x1FF) << 9);
}

// ============================================================================
// Device function: Store state (71 elements) to trace columns
// ============================================================================
__device__ void store_state_to_trace(
    m31** traces,
    uint32_t base_col,
    uint32_t idx,
    uint32_t table_offset,
    const m31* m_shifted,  // 14 elements
    const felt252& acc_x,
    const felt252& acc_y
) {
    // Column layout: table_offset(1) + m_shifted(14) + acc_x(28) + acc_y(28) = 71
    traces[base_col][idx] = {table_offset};

    for (int i = 0; i < 14; i++) {
        traces[base_col + 1 + i][idx] = m_shifted[i];
    }

    // Convert acc_x to 28 M31 limbs
    m31 x_limbs[28];
    felt252_to_28_limbs(acc_x, x_limbs);
    for (int i = 0; i < 28; i++) {
        traces[base_col + 15 + i][idx] = x_limbs[i];
    }

    // Convert acc_y to 28 M31 limbs
    m31 y_limbs[28];
    felt252_to_28_limbs(acc_y, y_limbs);
    for (int i = 0; i < 28; i++) {
        traces[base_col + 43 + i][idx] = y_limbs[i];
    }
}

// ============================================================================
// Full GPU base trace generation kernel for pedersen_builtin
// ============================================================================
__global__ void gen_pedersen_builtin_base_trace_full_gpu_kernel(
    m31** traces,
    uint32_t n_trace_columns,
    uint32_t n_rows,
    uint32_t segment_start,
    const m31* memory_address_to_id_table,
    const m31** memory_id_to_big_transposed,
    const m31* memory_id_to_big_small_values,
    // Lookup data outputs
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,
    m31** lookup_range_check_5_4_0,
    m31** lookup_range_check_5_4_1,
    m31** lookup_range_check_8_0,
    m31** lookup_range_check_8_1,
    m31** lookup_range_check_8_2,
    m31** lookup_range_check_8_3,
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_partial_ec_mul_2,
    m31** lookup_partial_ec_mul_3,
    m31** lookup_partial_ec_mul_4,
    m31** lookup_partial_ec_mul_5,
    m31** lookup_partial_ec_mul_6,
    m31** lookup_partial_ec_mul_7,
    // Sub-component inputs
    m31** sub_component_inputs_memory_address_to_id,
    m31** sub_component_inputs_memory_id_to_big
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    // Constants
    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_2 = {2};
    const m31 M31_3 = {3};
    const m31 M31_31 = {31};
    const m31 M31_32 = {32};
    const m31 M31_136 = {136};
    const m31 M31_256 = {256};
    const m31 M31_512 = {512};

    // Compute instance address: segment_start + idx * 3
    m31 seq = {idx};
    m31 segment_start_m31 = {segment_start};
    m31 instance_addr = add(segment_start_m31, mul(M31_3, seq));

    // ========================================================================
    // Read value A (first memory cell)
    // ========================================================================

    m31 pedersen_a_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr,
        &pedersen_a_id
    );

    m31 value_a_limbs[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        const_cast<m31**>(memory_id_to_big_transposed),
        const_cast<m31*>(memory_id_to_big_small_values),
        pedersen_a_id,
        value_a_limbs
    );

    // Store value A limbs in trace columns 0-26
    for (int i = 0; i < 27; i++) {
        traces[i][idx] = value_a_limbs[i];
    }

    // Compute ms_limb_low and ms_limb_high for value A (column 27)
    m31 ms_limb_a = value_a_limbs[27];
    m31 ms_limb_low_a = {ms_limb_a & 31};  // low 5 bits
    m31 ms_limb_high_a = {ms_limb_a >> 5}; // high 4 bits
    traces[27][idx] = ms_limb_low_a;
    traces[28][idx] = ms_limb_high_a;

    // Store pedersen_a_id in column 29
    traces[29][idx] = pedersen_a_id;

    // ========================================================================
    // Read value B (second memory cell)
    // ========================================================================

    m31 instance_addr_b = add(instance_addr, M31_1);
    m31 pedersen_b_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr_b,
        &pedersen_b_id
    );

    m31 value_b_limbs[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        const_cast<m31**>(memory_id_to_big_transposed),
        const_cast<m31*>(memory_id_to_big_small_values),
        pedersen_b_id,
        value_b_limbs
    );

    // Store value B limbs in trace columns 30-56
    for (int i = 0; i < 27; i++) {
        traces[30 + i][idx] = value_b_limbs[i];
    }

    // Compute ms_limb_low and ms_limb_high for value B
    m31 ms_limb_b = value_b_limbs[27];
    m31 ms_limb_low_b = {ms_limb_b & 31};
    m31 ms_limb_high_b = {ms_limb_b >> 5};
    traces[57][idx] = ms_limb_low_b;
    traces[58][idx] = ms_limb_high_b;

    // Store pedersen_b_id in column 59
    traces[59][idx] = pedersen_b_id;

    // ========================================================================
    // Range check columns (60-65)
    // ========================================================================

    m31 ms_limb_full_a = add(ms_limb_low_a, mul(ms_limb_high_a, M31_32));
    m31 ms_limb_is_max_a = {(ms_limb_full_a == 256) ? 1u : 0u};
    traces[60][idx] = ms_limb_is_max_a;

    m31 mid_limb_a = value_a_limbs[21];
    m31 ms_and_mid_limbs_are_max_a = {(ms_limb_is_max_a && mid_limb_a == 136) ? 1u : 0u};
    traces[61][idx] = ms_and_mid_limbs_are_max_a;

    const m31 M31_120 = {120};
    m31 rc_input_a = mul(ms_limb_is_max_a, sub(add(M31_120, mid_limb_a), ms_and_mid_limbs_are_max_a));
    traces[62][idx] = rc_input_a;

    m31 ms_limb_full_b = add(ms_limb_low_b, mul(ms_limb_high_b, M31_32));
    m31 ms_limb_is_max_b = {(ms_limb_full_b == 256) ? 1u : 0u};
    traces[63][idx] = ms_limb_is_max_b;

    m31 mid_limb_b = value_b_limbs[21];
    m31 ms_and_mid_limbs_are_max_b = {(ms_limb_is_max_b && mid_limb_b == 136) ? 1u : 0u};
    traces[64][idx] = ms_and_mid_limbs_are_max_b;

    m31 rc_input_b = mul(ms_limb_is_max_b, sub(add(M31_120, mid_limb_b), ms_and_mid_limbs_are_max_b));
    traces[65][idx] = rc_input_b;

    // ========================================================================
    // Compute partial_ec_mul on GPU (columns 66-349)
    // ========================================================================

    // Initialize m_shifted arrays for value A (14 18-bit windows from 252-bit value)
    m31 m_shifted_a[14];
    for (int i = 0; i < 13; i++) {
        // Each window is 18 bits = limb[2i] + limb[2i+1] * 512
        m_shifted_a[i] = add(value_a_limbs[2*i], mul(value_a_limbs[2*i + 1], M31_512));
    }
    // Last window uses ms_limb_low_a
    m_shifted_a[13] = add(value_a_limbs[26], mul(ms_limb_low_a, M31_512));

    // Initialize m_shifted arrays for value B
    m31 m_shifted_b[14];
    for (int i = 0; i < 13; i++) {
        m_shifted_b[i] = add(value_b_limbs[2*i], mul(value_b_limbs[2*i + 1], M31_512));
    }
    m_shifted_b[13] = add(value_b_limbs[26], mul(ms_limb_low_b, M31_512));

    // Initialize accumulator with shift point (convert from 28 M31 9-bit limbs to felt252)
    m31 shift_x_m31[28], shift_y_m31[28];
    for (int i = 0; i < 28; i++) {
        shift_x_m31[i] = {SHIFT_POINT_X_M31_LIMBS[i]};
        shift_y_m31[i] = {SHIFT_POINT_Y_M31_LIMBS[i]};
    }
    felt252 acc_x, acc_y;
    felt252_from_28_limbs(acc_x, shift_x_m31);
    felt252_from_28_limbs(acc_y, shift_y_m31);

    // Store initial state for partial_ec_mul lookup 0
    // lookup_partial_ec_mul_0: input to chain 1 round 0
    uint32_t row_chain_id = idx * 4;
    if (lookup_partial_ec_mul_0 != nullptr) {
        lookup_partial_ec_mul_0[0][idx] = {row_chain_id};      // chain_id
        lookup_partial_ec_mul_0[1][idx] = M31_0;               // round
        lookup_partial_ec_mul_0[2][idx] = {P0_TABLE_OFFSET};   // table_offset
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_0[3 + i][idx] = m_shifted_a[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_0[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_0[45 + i][idx] = y_limbs[i];
        }
    }

    // Chain 1: 14 rounds with P0 table (value A low bits)
    m31 chain1_m_shifted[14];
    for (int i = 0; i < 14; i++) chain1_m_shifted[i] = m_shifted_a[i];

    for (int round = 0; round < 14; round++) {
        uint32_t window_value = chain1_m_shifted[0];
        partial_ec_mul_round(P0_TABLE_OFFSET, round, window_value, chain1_m_shifted, acc_x, acc_y);
    }

    // Store chain 1 output (block 0: cols 66-136)
    store_state_to_trace(traces, 66, idx, P0_TABLE_OFFSET, chain1_m_shifted, acc_x, acc_y);

    // Store partial_ec_mul lookup 1: output of chain 1 at round 14
    if (lookup_partial_ec_mul_1 != nullptr) {
        lookup_partial_ec_mul_1[0][idx] = {row_chain_id};
        lookup_partial_ec_mul_1[1][idx] = {14};
        lookup_partial_ec_mul_1[2][idx] = {P0_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_1[3 + i][idx] = chain1_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_1[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_1[45 + i][idx] = y_limbs[i];
        }
    }

    // Chain 2: 1 round with P1 table (value A high bits - only 4 bits)
    m31 chain2_m_shifted[14];
    for (int i = 0; i < 14; i++) chain2_m_shifted[i] = M31_0;
    chain2_m_shifted[0] = ms_limb_high_a;

    // Store partial_ec_mul lookup 2: input to chain 2 round 0
    if (lookup_partial_ec_mul_2 != nullptr) {
        lookup_partial_ec_mul_2[0][idx] = {row_chain_id + 1};
        lookup_partial_ec_mul_2[1][idx] = M31_0;
        lookup_partial_ec_mul_2[2][idx] = {P1_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_2[3 + i][idx] = chain2_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_2[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_2[45 + i][idx] = y_limbs[i];
        }
    }

    uint32_t window_value_a_high = ms_limb_high_a;
    partial_ec_mul_round(P1_TABLE_OFFSET, 0, window_value_a_high, chain2_m_shifted, acc_x, acc_y);

    // Store chain 2 output (block 1: cols 137-207)
    store_state_to_trace(traces, 137, idx, P1_TABLE_OFFSET, chain2_m_shifted, acc_x, acc_y);

    // Store partial_ec_mul lookup 3: output of chain 2 at round 1
    if (lookup_partial_ec_mul_3 != nullptr) {
        lookup_partial_ec_mul_3[0][idx] = {row_chain_id + 1};
        lookup_partial_ec_mul_3[1][idx] = M31_1;
        lookup_partial_ec_mul_3[2][idx] = {P1_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_3[3 + i][idx] = chain2_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_3[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_3[45 + i][idx] = y_limbs[i];
        }
    }

    // Chain 3: 14 rounds with P2 table (value B low bits)
    m31 chain3_m_shifted[14];
    for (int i = 0; i < 14; i++) chain3_m_shifted[i] = m_shifted_b[i];

    // Store partial_ec_mul lookup 4: input to chain 3 round 0
    if (lookup_partial_ec_mul_4 != nullptr) {
        lookup_partial_ec_mul_4[0][idx] = {row_chain_id + 2};
        lookup_partial_ec_mul_4[1][idx] = M31_0;
        lookup_partial_ec_mul_4[2][idx] = {P2_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_4[3 + i][idx] = chain3_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_4[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_4[45 + i][idx] = y_limbs[i];
        }
    }

    for (int round = 0; round < 14; round++) {
        uint32_t window_value = chain3_m_shifted[0];
        partial_ec_mul_round(P2_TABLE_OFFSET, round, window_value, chain3_m_shifted, acc_x, acc_y);
    }

    // Store chain 3 output (block 2: cols 208-278)
    store_state_to_trace(traces, 208, idx, P2_TABLE_OFFSET, chain3_m_shifted, acc_x, acc_y);

    // Store partial_ec_mul lookup 5: output of chain 3 at round 14
    if (lookup_partial_ec_mul_5 != nullptr) {
        lookup_partial_ec_mul_5[0][idx] = {row_chain_id + 2};
        lookup_partial_ec_mul_5[1][idx] = {14};
        lookup_partial_ec_mul_5[2][idx] = {P2_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_5[3 + i][idx] = chain3_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_5[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_5[45 + i][idx] = y_limbs[i];
        }
    }

    // Chain 4: 1 round with P3 table (value B high bits - only 4 bits)
    m31 chain4_m_shifted[14];
    for (int i = 0; i < 14; i++) chain4_m_shifted[i] = M31_0;
    chain4_m_shifted[0] = ms_limb_high_b;

    // Store partial_ec_mul lookup 6: input to chain 4 round 0
    if (lookup_partial_ec_mul_6 != nullptr) {
        lookup_partial_ec_mul_6[0][idx] = {row_chain_id + 3};
        lookup_partial_ec_mul_6[1][idx] = M31_0;
        lookup_partial_ec_mul_6[2][idx] = {P3_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_6[3 + i][idx] = chain4_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_6[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_6[45 + i][idx] = y_limbs[i];
        }
    }

    uint32_t window_value_b_high = ms_limb_high_b;
    partial_ec_mul_round(P3_TABLE_OFFSET, 0, window_value_b_high, chain4_m_shifted, acc_x, acc_y);

    // Store chain 4 output (block 3: cols 279-349)
    store_state_to_trace(traces, 279, idx, P3_TABLE_OFFSET, chain4_m_shifted, acc_x, acc_y);

    // Store partial_ec_mul lookup 7: output of chain 4 after 1 round
    // Chain 4 does 1 round (round 0), so output is at "round 1" in the AIR's perspective
    if (lookup_partial_ec_mul_7 != nullptr) {
        lookup_partial_ec_mul_7[0][idx] = {row_chain_id + 3};
        lookup_partial_ec_mul_7[1][idx] = M31_1;  // round = 1 (after completing 1 round)
        lookup_partial_ec_mul_7[2][idx] = {P3_TABLE_OFFSET};
        for (int i = 0; i < 14; i++) {
            lookup_partial_ec_mul_7[3 + i][idx] = chain4_m_shifted[i];
        }
        m31 x_limbs[28], y_limbs[28];
        felt252_to_28_limbs(acc_x, x_limbs);
        felt252_to_28_limbs(acc_y, y_limbs);
        for (int i = 0; i < 28; i++) {
            lookup_partial_ec_mul_7[17 + i][idx] = x_limbs[i];
            lookup_partial_ec_mul_7[45 + i][idx] = y_limbs[i];
        }
    }

    // ========================================================================
    // Populate lookup data for memory_address_to_id
    // ========================================================================

    lookup_memory_address_to_id_0[0][idx] = instance_addr;
    lookup_memory_address_to_id_0[1][idx] = pedersen_a_id;

    lookup_memory_address_to_id_1[0][idx] = instance_addr_b;
    lookup_memory_address_to_id_1[1][idx] = pedersen_b_id;

    m31 instance_addr_c = add(instance_addr, M31_2);
    m31 pedersen_result_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr_c,
        &pedersen_result_id
    );
    traces[350][idx] = pedersen_result_id;
    lookup_memory_address_to_id_2[0][idx] = instance_addr_c;
    lookup_memory_address_to_id_2[1][idx] = pedersen_result_id;

    // ========================================================================
    // Populate lookup data for memory_id_to_big
    // ========================================================================

    lookup_memory_id_to_big_0[0][idx] = pedersen_a_id;
    for (int i = 0; i < 27; i++) {
        lookup_memory_id_to_big_0[1 + i][idx] = value_a_limbs[i];
    }
    lookup_memory_id_to_big_0[28][idx] = ms_limb_full_a;

    lookup_memory_id_to_big_1[0][idx] = pedersen_b_id;
    for (int i = 0; i < 27; i++) {
        lookup_memory_id_to_big_1[1 + i][idx] = value_b_limbs[i];
    }
    lookup_memory_id_to_big_1[28][idx] = ms_limb_full_b;

    // Result value from final accumulator (chain4 output)
    lookup_memory_id_to_big_2[0][idx] = pedersen_result_id;
    m31 result_x_limbs[28];
    felt252_to_28_limbs(acc_x, result_x_limbs);
    for (int i = 0; i < 28; i++) {
        lookup_memory_id_to_big_2[1 + i][idx] = result_x_limbs[i];
    }

    // ========================================================================
    // Populate lookup data for range checks
    // ========================================================================

    lookup_range_check_5_4_0[0][idx] = ms_limb_low_a;
    lookup_range_check_5_4_0[1][idx] = ms_limb_high_a;
    lookup_range_check_5_4_1[0][idx] = ms_limb_low_b;
    lookup_range_check_5_4_1[1][idx] = ms_limb_high_b;

    lookup_range_check_8_0[0][idx] = sub(ms_limb_full_a, ms_limb_is_max_a);
    lookup_range_check_8_1[0][idx] = rc_input_a;
    lookup_range_check_8_2[0][idx] = sub(ms_limb_full_b, ms_limb_is_max_b);
    lookup_range_check_8_3[0][idx] = rc_input_b;

    // ========================================================================
    // Populate sub-component inputs
    // ========================================================================

    sub_component_inputs_memory_address_to_id[0][idx] = instance_addr;
    sub_component_inputs_memory_address_to_id[1][idx] = instance_addr_b;
    sub_component_inputs_memory_address_to_id[2][idx] = instance_addr_c;

    sub_component_inputs_memory_id_to_big[0][idx] = pedersen_a_id;
    sub_component_inputs_memory_id_to_big[1][idx] = pedersen_b_id;
    sub_component_inputs_memory_id_to_big[2][idx] = pedersen_result_id;
}

// ============================================================================
// Original base trace generation kernel (hybrid CPU/GPU approach)
// ============================================================================

__global__ void gen_pedersen_builtin_base_trace_kernel(
    m31** traces,
    uint32_t n_trace_columns,
    uint32_t n_rows,
    uint32_t segment_start,
    const m31* memory_address_to_id_table,
    const m31** memory_id_to_big_transposed,
    const m31* memory_id_to_big_small_values,
    // Pre-computed EC columns (columns 66-350)
    const m31** precomputed_ec_columns,
    uint32_t n_precomputed_ec_columns,
    // Lookup data outputs
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,
    m31** lookup_range_check_5_4_0,
    m31** lookup_range_check_5_4_1,
    m31** lookup_range_check_8_0,
    m31** lookup_range_check_8_1,
    m31** lookup_range_check_8_2,
    m31** lookup_range_check_8_3,
    // Sub-component inputs
    m31** sub_component_inputs_memory_address_to_id,
    m31** sub_component_inputs_memory_id_to_big
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    // Constants
    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_2 = {2};
    const m31 M31_3 = {3};
    const m31 M31_31 = {31};
    const m31 M31_32 = {32};
    const m31 M31_136 = {136};
    const m31 M31_256 = {256};

    // Compute instance address: segment_start + idx * 3
    m31 seq = {idx};
    m31 segment_start_m31 = {segment_start};
    m31 instance_addr = add(segment_start_m31, mul(M31_3, seq));

    // ========================================================================
    // Read value A (first memory cell)
    // ========================================================================

    // Memory address to ID lookup
    m31 pedersen_a_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr,
        &pedersen_a_id
    );

    // Memory ID to big lookup - get 28 limbs
    m31 value_a_limbs[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        const_cast<m31**>(memory_id_to_big_transposed),
        const_cast<m31*>(memory_id_to_big_small_values),
        pedersen_a_id,
        value_a_limbs
    );

    // Store value A limbs in trace columns 0-26
    for (int i = 0; i < 27; i++) {
        traces[i][idx] = value_a_limbs[i];
    }

    // Compute ms_limb_low and ms_limb_high for value A (column 27)
    m31 ms_limb_a = value_a_limbs[27];
    m31 ms_limb_low_a = {ms_limb_a & 31};  // low 5 bits
    m31 ms_limb_high_a = {ms_limb_a >> 5}; // high 4 bits
    traces[27][idx] = ms_limb_low_a;
    traces[28][idx] = ms_limb_high_a;

    // Store pedersen_a_id in column 29
    traces[29][idx] = pedersen_a_id;

    // ========================================================================
    // Read value B (second memory cell)
    // ========================================================================

    m31 instance_addr_b = add(instance_addr, M31_1);
    m31 pedersen_b_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr_b,
        &pedersen_b_id
    );

    // Memory ID to big lookup for value B
    m31 value_b_limbs[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        const_cast<m31**>(memory_id_to_big_transposed),
        const_cast<m31*>(memory_id_to_big_small_values),
        pedersen_b_id,
        value_b_limbs
    );

    // Store value B limbs in trace columns 30-56
    for (int i = 0; i < 27; i++) {
        traces[30 + i][idx] = value_b_limbs[i];
    }

    // Compute ms_limb_low and ms_limb_high for value B
    m31 ms_limb_b = value_b_limbs[27];
    m31 ms_limb_low_b = {ms_limb_b & 31};
    m31 ms_limb_high_b = {ms_limb_b >> 5};
    traces[57][idx] = ms_limb_low_b;
    traces[58][idx] = ms_limb_high_b;

    // Store pedersen_b_id in column 59
    traces[59][idx] = pedersen_b_id;

    // ========================================================================
    // Range check columns (60-65)
    // ========================================================================

    // ms_limb_is_max for A: check if ms_limb equals 256 (max value)
    m31 ms_limb_full_a = add(ms_limb_low_a, mul(ms_limb_high_a, M31_32));
    m31 ms_limb_is_max_a = {(ms_limb_full_a == 256) ? 1u : 0u};
    traces[60][idx] = ms_limb_is_max_a;

    // ms_and_mid_limbs_are_max for A: ms_limb==256 && value_limb_21==136
    m31 mid_limb_a = value_a_limbs[21];
    m31 ms_and_mid_limbs_are_max_a = {(ms_limb_is_max_a && mid_limb_a == 136) ? 1u : 0u};
    traces[61][idx] = ms_and_mid_limbs_are_max_a;

    // rc_input for A (column 62): ms_limb_is_max * (120 + value_limb_21 - ms_and_mid_limbs_are_max)
    const m31 M31_120 = {120};
    m31 rc_input_a = mul(ms_limb_is_max_a, sub(add(M31_120, mid_limb_a), ms_and_mid_limbs_are_max_a));
    traces[62][idx] = rc_input_a;

    // ms_limb_is_max for B
    m31 ms_limb_full_b = add(ms_limb_low_b, mul(ms_limb_high_b, M31_32));
    m31 ms_limb_is_max_b = {(ms_limb_full_b == 256) ? 1u : 0u};
    traces[63][idx] = ms_limb_is_max_b;

    // ms_and_mid_limbs_are_max for B: ms_limb==256 && value_limb_21==136
    m31 mid_limb_b = value_b_limbs[21];
    m31 ms_and_mid_limbs_are_max_b = {(ms_limb_is_max_b && mid_limb_b == 136) ? 1u : 0u};
    traces[64][idx] = ms_and_mid_limbs_are_max_b;

    // rc_input for B (column 65): ms_limb_is_max * (120 + value_limb_21 - ms_and_mid_limbs_are_max)
    m31 rc_input_b = mul(ms_limb_is_max_b, sub(add(M31_120, mid_limb_b), ms_and_mid_limbs_are_max_b));
    traces[65][idx] = rc_input_b;

    // ========================================================================
    // Copy pre-computed EC columns (66-350)
    // ========================================================================

    for (uint32_t i = 0; i < n_precomputed_ec_columns; i++) {
        traces[N_MEMORY_COLUMNS + i][idx] = precomputed_ec_columns[i][idx];
    }

    // ========================================================================
    // Populate lookup data for memory_address_to_id
    // ========================================================================

    // Lookup 0: (instance_addr, pedersen_a_id)
    lookup_memory_address_to_id_0[0][idx] = instance_addr;
    lookup_memory_address_to_id_0[1][idx] = pedersen_a_id;

    // Lookup 1: (instance_addr + 1, pedersen_b_id)
    lookup_memory_address_to_id_1[0][idx] = instance_addr_b;
    lookup_memory_address_to_id_1[1][idx] = pedersen_b_id;

    // Lookup 2: (instance_addr + 2, pedersen_result_id)
    // The result ID is stored in column 350
    m31 instance_addr_c = add(instance_addr, M31_2);
    m31 pedersen_result_id = {0};
    memory_address_to_id_deduce_output(
        const_cast<m31*>(memory_address_to_id_table),
        instance_addr_c,
        &pedersen_result_id
    );
    // Store pedersen_result_id in column 350
    traces[350][idx] = pedersen_result_id;
    lookup_memory_address_to_id_2[0][idx] = instance_addr_c;
    lookup_memory_address_to_id_2[1][idx] = pedersen_result_id;

    // ========================================================================
    // Populate lookup data for memory_id_to_big
    // ========================================================================

    // Lookup 0: [pedersen_a_id, 28 value limbs]
    lookup_memory_id_to_big_0[0][idx] = pedersen_a_id;
    for (int i = 0; i < 27; i++) {
        lookup_memory_id_to_big_0[1 + i][idx] = value_a_limbs[i];
    }
    lookup_memory_id_to_big_0[28][idx] = ms_limb_full_a;

    // Lookup 1: [pedersen_b_id, 28 value limbs]
    lookup_memory_id_to_big_1[0][idx] = pedersen_b_id;
    for (int i = 0; i < 27; i++) {
        lookup_memory_id_to_big_1[1 + i][idx] = value_b_limbs[i];
    }
    lookup_memory_id_to_big_1[28][idx] = ms_limb_full_b;

    // Lookup 2: [pedersen_result_id, result value limbs]
    // The result value comes from the final EC accumulator in precomputed columns
    lookup_memory_id_to_big_2[0][idx] = pedersen_result_id;
    // Result value limbs are in precomputed columns - need to map appropriately
    // The final accumulator x-coordinate limbs start at column index 213+15 = 228 in precomputed
    for (int i = 0; i < 28; i++) {
        // Get result from final chain output
        uint32_t precomputed_idx = 213 + 15 + i; // block 3 acc_x start
        if (precomputed_idx < n_precomputed_ec_columns) {
            lookup_memory_id_to_big_2[1 + i][idx] = precomputed_ec_columns[precomputed_idx][idx];
        } else {
            lookup_memory_id_to_big_2[1 + i][idx] = M31_0;
        }
    }

    // ========================================================================
    // Populate lookup data for range checks
    // ========================================================================

    // range_check_5_4 lookups
    lookup_range_check_5_4_0[0][idx] = ms_limb_low_a;
    lookup_range_check_5_4_0[1][idx] = ms_limb_high_a;
    lookup_range_check_5_4_1[0][idx] = ms_limb_low_b;
    lookup_range_check_5_4_1[1][idx] = ms_limb_high_b;

    // range_check_8 lookups
    // range_check_8_0: ms_limb_full_a - ms_limb_is_max_a
    lookup_range_check_8_0[0][idx] = sub(ms_limb_full_a, ms_limb_is_max_a);
    // range_check_8_1: rc_input_col62
    lookup_range_check_8_1[0][idx] = rc_input_a;
    // range_check_8_2: ms_limb_full_b - ms_limb_is_max_b
    lookup_range_check_8_2[0][idx] = sub(ms_limb_full_b, ms_limb_is_max_b);
    // range_check_8_3: rc_input_col65
    lookup_range_check_8_3[0][idx] = rc_input_b;

    // ========================================================================
    // Populate sub-component inputs
    // ========================================================================

    // memory_address_to_id inputs (3 inputs)
    sub_component_inputs_memory_address_to_id[0][idx] = instance_addr;
    sub_component_inputs_memory_address_to_id[1][idx] = instance_addr_b;
    sub_component_inputs_memory_address_to_id[2][idx] = instance_addr_c;

    // memory_id_to_big inputs (3 inputs)
    sub_component_inputs_memory_id_to_big[0][idx] = pedersen_a_id;
    sub_component_inputs_memory_id_to_big[1][idx] = pedersen_b_id;
    sub_component_inputs_memory_id_to_big[2][idx] = pedersen_result_id;
}

// ============================================================================
// Interaction trace generation kernel for pedersen_builtin
// ============================================================================

// ============================================================================
// Template kernel for logup column generation (pair of lookups, addition)
// ============================================================================
template <int N, int M>
__global__ void gen_pedersen_interaction_col_gen_add_kernel(
    LookupElementsBasic<N>* lookup_elements_n,
    LookupElementsBasic<M>* lookup_elements_m,
    m31** lookup_state_0,
    m31** lookup_state_1,
    uint32_t trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    m31 values0[N];
    m31 values1[M];

    for (int i = 0; i < N; i++) {
        values0[i] = lookup_state_0[i][idx];
    }
    for (int i = 0; i < M; i++) {
        values1[i] = lookup_state_1[i][idx];
    }

    qm31 denom0 = lookup_elements_n->combine(values0, N);
    qm31 denom1 = lookup_elements_m->combine(values1, M);

    // For addition: numerator = denom0 + denom1
    logup_col_write_frac(idx, add(denom0, denom1), mul(denom0, denom1),
                         denom_ptr, numerator0, numerator1, numerator2, numerator3);
}

// ============================================================================
// Template kernel for logup column generation (pair of lookups, subtraction)
// ============================================================================
template <int N, int M>
__global__ void gen_pedersen_interaction_col_gen_sub_kernel(
    LookupElementsBasic<N>* lookup_elements_n,
    LookupElementsBasic<M>* lookup_elements_m,
    m31** lookup_state_0,
    m31** lookup_state_1,
    uint32_t trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    m31 values0[N];
    m31 values1[M];

    for (int i = 0; i < N; i++) {
        values0[i] = lookup_state_0[i][idx];
    }
    for (int i = 0; i < M; i++) {
        values1[i] = lookup_state_1[i][idx];
    }

    qm31 denom0 = lookup_elements_n->combine(values0, N);
    qm31 denom1 = lookup_elements_m->combine(values1, M);

    // For subtraction: numerator = denom0 - denom1
    logup_col_write_frac(idx, sub(denom0, denom1), mul(denom0, denom1),
                         denom_ptr, numerator0, numerator1, numerator2, numerator3);
}

// ============================================================================
// Finalize column kernel - accumulates interaction trace values
// Following the pattern from add_opcode: accumulate all fractions into running sum
// ============================================================================
__global__ void gen_pedersen_interaction_finalize_col_kernel(
    uint32_t col_index,
    uint32_t trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    int pre_index = (col_index == 0) ? -1 : static_cast<int>(col_index - 1);

    qm31 value = mul(
        qm31 {
            cm31{numerator0[idx], numerator1[idx]},
            cm31{numerator2[idx], numerator3[idx]}
        },
        denom_inv_ptr[idx]
    );

    if (pre_index == -1) {
        // First column - just store the value
        qm31 tmp = value;
        numerator0[idx] = tmp.a.a;
        numerator1[idx] = tmp.a.b;
        numerator2[idx] = tmp.b.a;
        numerator3[idx] = tmp.b.b;
    } else {
        // Add to previous column value
        qm31 prev_value = qm31 {
            cm31{interaction_traces[pre_index * 4 + 0][idx],
                 interaction_traces[pre_index * 4 + 1][idx]},
            cm31{interaction_traces[pre_index * 4 + 2][idx],
                 interaction_traces[pre_index * 4 + 3][idx]}
        };
        qm31 tmp = add(value, prev_value);
        numerator0[idx] = tmp.a.a;
        numerator1[idx] = tmp.a.b;
        numerator2[idx] = tmp.b.a;
        numerator3[idx] = tmp.b.b;
    }

    // Write accumulated sum to interaction_traces for this column
    interaction_traces[col_index * 4 + 0][idx] = numerator0[idx];
    interaction_traces[col_index * 4 + 1][idx] = numerator1[idx];
    interaction_traces[col_index * 4 + 2][idx] = numerator2[idx];
    interaction_traces[col_index * 4 + 3][idx] = numerator3[idx];
}

// ============================================================================
// Cumsum shift kernel - computes claimed_sum from last column
// ============================================================================
__global__ void gen_pedersen_interaction_cumsum_shift_kernel(
    uint32_t last_index,
    uint32_t trace_size,
    m31** interaction_traces,
    m31* coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = {0};
    m31 sum1 = {0};
    m31 sum2 = {0};
    m31 sum3 = {0};

    for (uint32_t i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[idx0][i]);
        sum1 = add(sum1, interaction_traces[idx1][i]);
        sum2 = add(sum2, interaction_traces[idx2][i]);
        sum3 = add(sum3, interaction_traces[idx3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

// ============================================================================
// Coordinate prefix sum kernel - subtracts claimed_sum/trace_size from last column
// ============================================================================
__global__ void gen_pedersen_interaction_coord_prefix_sum_kernel(
    m31* coordinate_sums,
    uint32_t last_index,
    uint32_t trace_size,
    m31** interaction_traces
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    qm31 claimed_sum = qm31 {
        cm31{coordinate_sums[0], coordinate_sums[1]},
        cm31{coordinate_sums[2], coordinate_sums[3]}
    };
    qm31 cumsum_shift = div(claimed_sum, m31{trace_size});

    interaction_traces[4 * last_index - 4][idx] = sub(interaction_traces[4 * last_index - 4][idx], cumsum_shift.a.a);
    interaction_traces[4 * last_index - 3][idx] = sub(interaction_traces[4 * last_index - 3][idx], cumsum_shift.a.b);
    interaction_traces[4 * last_index - 2][idx] = sub(interaction_traces[4 * last_index - 2][idx], cumsum_shift.b.a);
    interaction_traces[4 * last_index - 1][idx] = sub(interaction_traces[4 * last_index - 1][idx], cumsum_shift.b.b);
}

// ============================================================================
// Compute claimed sum kernel
// ============================================================================
__global__ void gen_pedersen_compute_claimed_sum_kernel(
    uint32_t n_cols,
    uint32_t trace_size,
    m31** interaction_traces,
    qm31* claimed_sum
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t last_idx = trace_size - 1;
        uint32_t last_col = n_cols - 1;

        qm31 sum = qm31 {
            cm31{interaction_traces[last_col * 4 + 0][last_idx],
                 interaction_traces[last_col * 4 + 1][last_idx]},
            cm31{interaction_traces[last_col * 4 + 2][last_idx],
                 interaction_traces[last_col * 4 + 3][last_idx]}
        };
        *claimed_sum = sum;
    }
}

// ============================================================================
// Reduction kernel for computing claimed_sum
// ============================================================================

__global__ void reduce_claimed_sum_kernel(
    const qm31* partial_sums,
    uint32_t n_elements,
    qm31* claimed_sum
) {
    // Simple reduction - for production use tree reduction
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        qm31 sum = {0, 0, 0, 0};
        for (uint32_t i = 0; i < n_elements; i++) {
            sum = add(sum, partial_sums[i]);
        }
        *claimed_sum = sum;
    }
}

// ============================================================================
// C API - Pedersen Table Management
// ============================================================================

extern "C" void pedersen_table_init(m31** columns, uint32_t n_rows) {
    pedersen_table_init_impl(columns, n_rows);
}

extern "C" void pedersen_table_free() {
    pedersen_table_free_impl();
}

// ============================================================================
// C API - Original Base Trace Generation (hybrid CPU/GPU)
// ============================================================================

extern "C" void gen_pedersen_builtin_trace(
    m31** traces,
    uint32_t n_trace_columns,

    // Lookup data - memory_address_to_id (3 lookups)
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,

    // Lookup data - memory_id_to_big (3 lookups)
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,

    // Lookup data - partial_ec_mul (8 lookups)
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_partial_ec_mul_2,
    m31** lookup_partial_ec_mul_3,
    m31** lookup_partial_ec_mul_4,
    m31** lookup_partial_ec_mul_5,
    m31** lookup_partial_ec_mul_6,
    m31** lookup_partial_ec_mul_7,

    // Lookup data - range_check_5_4 (2 lookups)
    m31** lookup_range_check_5_4_0,
    m31** lookup_range_check_5_4_1,

    // Lookup data - range_check_8 (4 lookups)
    m31** lookup_range_check_8_0,
    m31** lookup_range_check_8_1,
    m31** lookup_range_check_8_2,
    m31** lookup_range_check_8_3,

    // Sub-component inputs
    m31** sub_component_inputs_memory_address_to_id,
    m31** sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    uint32_t segment_start,

    // Memory lookup tables
    m31* memory_address_to_id_address_to_raw_id,
    m31** memory_id_to_big_transposed_big_values,
    m31* memory_id_to_big_small_values,

    // Pre-computed EC columns (columns 66-350)
    m31** precomputed_ec_columns,
    uint32_t n_precomputed_ec_columns,

    uint32_t n_rows,
    uint32_t log_size
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    timer global_timer;
    global_timer.start("generate pedersen_builtin base trace");
    // Clone pointer arrays to device memory
    m31** device_traces = clone_to_device<m31*>(traces, n_trace_columns);
    m31** device_memory_id_to_big_transposed = clone_to_device<m31*>(memory_id_to_big_transposed_big_values, 29);
    m31** device_precomputed_ec = clone_to_device<m31*>(precomputed_ec_columns, n_precomputed_ec_columns);

    // Clone lookup data pointer arrays
    m31** device_lookup_addr2id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31** device_lookup_addr2id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31** device_lookup_addr2id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31** device_lookup_id2big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31** device_lookup_id2big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31** device_lookup_id2big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31** device_lookup_rc54_0 = clone_to_device<m31*>(lookup_range_check_5_4_0, 2);
    m31** device_lookup_rc54_1 = clone_to_device<m31*>(lookup_range_check_5_4_1, 2);
    m31** device_lookup_rc8_0 = clone_to_device<m31*>(lookup_range_check_8_0, 1);
    m31** device_lookup_rc8_1 = clone_to_device<m31*>(lookup_range_check_8_1, 1);
    m31** device_lookup_rc8_2 = clone_to_device<m31*>(lookup_range_check_8_2, 1);
    m31** device_lookup_rc8_3 = clone_to_device<m31*>(lookup_range_check_8_3, 1);

    // Clone sub-component input pointer arrays
    m31** device_sub_addr2id = clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 3);
    m31** device_sub_id2big = clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 3);

    gen_pedersen_builtin_base_trace_kernel<<<num_blocks, BLOCK_SIZE>>>(
        device_traces,
        n_trace_columns,
        n_rows,
        segment_start,
        memory_address_to_id_address_to_raw_id,
        const_cast<const m31**>(device_memory_id_to_big_transposed),
        memory_id_to_big_small_values,
        const_cast<const m31**>(device_precomputed_ec),
        n_precomputed_ec_columns,
        // Lookup data outputs
        device_lookup_addr2id_0,
        device_lookup_addr2id_1,
        device_lookup_addr2id_2,
        device_lookup_id2big_0,
        device_lookup_id2big_1,
        device_lookup_id2big_2,
        device_lookup_rc54_0,
        device_lookup_rc54_1,
        device_lookup_rc8_0,
        device_lookup_rc8_1,
        device_lookup_rc8_2,
        device_lookup_rc8_3,
        // Sub-component inputs
        device_sub_addr2id,
        device_sub_id2big
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("generate pedersen_builtin base trace");

    // Free device memory for pointer arrays
    cuda_free_memory(device_traces);
    cuda_free_memory(device_memory_id_to_big_transposed);
    cuda_free_memory(device_precomputed_ec);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_rc54_0);
    cuda_free_memory(device_lookup_rc54_1);
    cuda_free_memory(device_lookup_rc8_0);
    cuda_free_memory(device_lookup_rc8_1);
    cuda_free_memory(device_lookup_rc8_2);
    cuda_free_memory(device_lookup_rc8_3);
    cuda_free_memory(device_sub_addr2id);
    cuda_free_memory(device_sub_id2big);
}

// ============================================================================
// C API - Full GPU Base Trace Generation (computes partial_ec_mul on GPU)
// ============================================================================

extern "C" void gen_pedersen_builtin_trace_full_gpu(
    m31** traces,
    uint32_t n_trace_columns,

    // Lookup data - memory_address_to_id (3 lookups)
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,

    // Lookup data - memory_id_to_big (3 lookups)
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,

    // Lookup data - partial_ec_mul (8 lookups)
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_partial_ec_mul_2,
    m31** lookup_partial_ec_mul_3,
    m31** lookup_partial_ec_mul_4,
    m31** lookup_partial_ec_mul_5,
    m31** lookup_partial_ec_mul_6,
    m31** lookup_partial_ec_mul_7,

    // Lookup data - range_check_5_4 (2 lookups)
    m31** lookup_range_check_5_4_0,
    m31** lookup_range_check_5_4_1,

    // Lookup data - range_check_8 (4 lookups)
    m31** lookup_range_check_8_0,
    m31** lookup_range_check_8_1,
    m31** lookup_range_check_8_2,
    m31** lookup_range_check_8_3,

    // Sub-component inputs
    m31** sub_component_inputs_memory_address_to_id,
    m31** sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    uint32_t segment_start,

    // Memory lookup tables
    m31* memory_address_to_id_address_to_raw_id,
    m31** memory_id_to_big_transposed_big_values,
    m31* memory_id_to_big_small_values,

    uint32_t n_rows,
    uint32_t log_size
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    timer global_timer;
    global_timer.start("generate pedersen_builtin base trace (full GPU)");

    // Clone pointer arrays to device memory
    m31** device_traces = clone_to_device<m31*>(traces, n_trace_columns);
    m31** device_memory_id_to_big_transposed = clone_to_device<m31*>(memory_id_to_big_transposed_big_values, 29);

    // Clone lookup data pointer arrays
    m31** device_lookup_addr2id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31** device_lookup_addr2id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31** device_lookup_addr2id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31** device_lookup_id2big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31** device_lookup_id2big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31** device_lookup_id2big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31** device_lookup_rc54_0 = clone_to_device<m31*>(lookup_range_check_5_4_0, 2);
    m31** device_lookup_rc54_1 = clone_to_device<m31*>(lookup_range_check_5_4_1, 2);
    m31** device_lookup_rc8_0 = clone_to_device<m31*>(lookup_range_check_8_0, 1);
    m31** device_lookup_rc8_1 = clone_to_device<m31*>(lookup_range_check_8_1, 1);
    m31** device_lookup_rc8_2 = clone_to_device<m31*>(lookup_range_check_8_2, 1);
    m31** device_lookup_rc8_3 = clone_to_device<m31*>(lookup_range_check_8_3, 1);
    m31** device_lookup_ec_mul_0 = clone_to_device<m31*>(lookup_partial_ec_mul_0, 73);
    m31** device_lookup_ec_mul_1 = clone_to_device<m31*>(lookup_partial_ec_mul_1, 73);
    m31** device_lookup_ec_mul_2 = clone_to_device<m31*>(lookup_partial_ec_mul_2, 73);
    m31** device_lookup_ec_mul_3 = clone_to_device<m31*>(lookup_partial_ec_mul_3, 73);
    m31** device_lookup_ec_mul_4 = clone_to_device<m31*>(lookup_partial_ec_mul_4, 73);
    m31** device_lookup_ec_mul_5 = clone_to_device<m31*>(lookup_partial_ec_mul_5, 73);
    m31** device_lookup_ec_mul_6 = clone_to_device<m31*>(lookup_partial_ec_mul_6, 73);
    m31** device_lookup_ec_mul_7 = clone_to_device<m31*>(lookup_partial_ec_mul_7, 73);

    // Clone sub-component input pointer arrays
    m31** device_sub_addr2id = clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 3);
    m31** device_sub_id2big = clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 3);

    gen_pedersen_builtin_base_trace_full_gpu_kernel<<<num_blocks, BLOCK_SIZE>>>(
        device_traces,
        n_trace_columns,
        n_rows,
        segment_start,
        memory_address_to_id_address_to_raw_id,
        const_cast<const m31**>(device_memory_id_to_big_transposed),
        memory_id_to_big_small_values,
        // Lookup data outputs
        device_lookup_addr2id_0,
        device_lookup_addr2id_1,
        device_lookup_addr2id_2,
        device_lookup_id2big_0,
        device_lookup_id2big_1,
        device_lookup_id2big_2,
        device_lookup_rc54_0,
        device_lookup_rc54_1,
        device_lookup_rc8_0,
        device_lookup_rc8_1,
        device_lookup_rc8_2,
        device_lookup_rc8_3,
        device_lookup_ec_mul_0,
        device_lookup_ec_mul_1,
        device_lookup_ec_mul_2,
        device_lookup_ec_mul_3,
        device_lookup_ec_mul_4,
        device_lookup_ec_mul_5,
        device_lookup_ec_mul_6,
        device_lookup_ec_mul_7,
        // Sub-component inputs
        device_sub_addr2id,
        device_sub_id2big
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("generate pedersen_builtin base trace (full GPU)");

    // Free device memory for pointer arrays
    cuda_free_memory(device_traces);
    cuda_free_memory(device_memory_id_to_big_transposed);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_rc54_0);
    cuda_free_memory(device_lookup_rc54_1);
    cuda_free_memory(device_lookup_rc8_0);
    cuda_free_memory(device_lookup_rc8_1);
    cuda_free_memory(device_lookup_rc8_2);
    cuda_free_memory(device_lookup_rc8_3);
    cuda_free_memory(device_lookup_ec_mul_0);
    cuda_free_memory(device_lookup_ec_mul_1);
    cuda_free_memory(device_lookup_ec_mul_2);
    cuda_free_memory(device_lookup_ec_mul_3);
    cuda_free_memory(device_lookup_ec_mul_4);
    cuda_free_memory(device_lookup_ec_mul_5);
    cuda_free_memory(device_lookup_ec_mul_6);
    cuda_free_memory(device_lookup_ec_mul_7);
    cuda_free_memory(device_sub_addr2id);
    cuda_free_memory(device_sub_id2big);
}

// ============================================================================
// C API - Interaction Trace Generation
// ============================================================================

extern "C" void gen_pedersen_builtin_interaction_trace(
    m31** interaction_trace,
    uint32_t n_interaction_columns,

    // Lookup data - memory_address_to_id (3 lookups)
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,

    // Lookup data - memory_id_to_big (3 lookups)
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,

    // Lookup data - partial_ec_mul (8 lookups)
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_partial_ec_mul_2,
    m31** lookup_partial_ec_mul_3,
    m31** lookup_partial_ec_mul_4,
    m31** lookup_partial_ec_mul_5,
    m31** lookup_partial_ec_mul_6,
    m31** lookup_partial_ec_mul_7,

    // Lookup data - range_check_5_4 (2 lookups)
    m31** lookup_range_check_5_4_0,
    m31** lookup_range_check_5_4_1,

    // Lookup data - range_check_8 (4 lookups)
    m31** lookup_range_check_8_0,
    m31** lookup_range_check_8_1,
    m31** lookup_range_check_8_2,
    m31** lookup_range_check_8_3,

    // Lookup elements (relations) - each is a LookupElementsBasic struct
    const uint32_t* memory_address_to_id_relation,
    const uint32_t* memory_id_to_big_relation,
    const uint32_t* partial_ec_mul_relation,
    const uint32_t* range_check_5_4_relation,
    const uint32_t* range_check_8_relation,

    m31* claimed_sum,
    uint32_t n_rows,
    uint32_t log_size
) {
    timer global_timer;
    global_timer.start("generate pedersen_builtin interaction trace");

    const int BLOCK_SIZE = 256;
    int num_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint32_t N_LOGUP_COLS = 10;  // 10 logup columns -> 40 M31 columns

    // Clone pointer arrays to device memory
    m31** device_interaction_trace = clone_to_device<m31*>(interaction_trace, n_interaction_columns);
    m31** device_lookup_addr2id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31** device_lookup_addr2id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31** device_lookup_addr2id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31** device_lookup_id2big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31** device_lookup_id2big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31** device_lookup_id2big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31** device_lookup_ec_mul_0 = clone_to_device<m31*>(lookup_partial_ec_mul_0, 73);
    m31** device_lookup_ec_mul_1 = clone_to_device<m31*>(lookup_partial_ec_mul_1, 73);
    m31** device_lookup_ec_mul_2 = clone_to_device<m31*>(lookup_partial_ec_mul_2, 73);
    m31** device_lookup_ec_mul_3 = clone_to_device<m31*>(lookup_partial_ec_mul_3, 73);
    m31** device_lookup_ec_mul_4 = clone_to_device<m31*>(lookup_partial_ec_mul_4, 73);
    m31** device_lookup_ec_mul_5 = clone_to_device<m31*>(lookup_partial_ec_mul_5, 73);
    m31** device_lookup_ec_mul_6 = clone_to_device<m31*>(lookup_partial_ec_mul_6, 73);
    m31** device_lookup_ec_mul_7 = clone_to_device<m31*>(lookup_partial_ec_mul_7, 73);
    m31** device_lookup_rc54_0 = clone_to_device<m31*>(lookup_range_check_5_4_0, 2);
    m31** device_lookup_rc54_1 = clone_to_device<m31*>(lookup_range_check_5_4_1, 2);
    m31** device_lookup_rc8_0 = clone_to_device<m31*>(lookup_range_check_8_0, 1);
    m31** device_lookup_rc8_1 = clone_to_device<m31*>(lookup_range_check_8_1, 1);
    m31** device_lookup_rc8_2 = clone_to_device<m31*>(lookup_range_check_8_2, 1);
    m31** device_lookup_rc8_3 = clone_to_device<m31*>(lookup_range_check_8_3, 1);

    // Allocate temporary buffers for logup computation
    qm31* d_denom;
    qm31* d_denom_inv;
    m31* d_numerator0;
    m31* d_numerator1;
    m31* d_numerator2;
    m31* d_numerator3;
    d_denom = cuda_malloc<qm31>(n_rows);
    d_denom_inv = cuda_malloc<qm31>(n_rows);
    d_numerator0 = cuda_malloc<m31>(n_rows);
    d_numerator1 = cuda_malloc<m31>(n_rows);
    d_numerator2 = cuda_malloc<m31>(n_rows);
    d_numerator3 = cuda_malloc<m31>(n_rows);

    // Clone relation structs to device
    // MemoryAddressToId: LookupElementsBasic<2> - z(1 qm31) + alpha(1 qm31) + alpha_powers[2](2 qm31) = 4 qm31 = 16 m31
    // MemoryIdToBig: LookupElementsBasic<29> - z + alpha + alpha_powers[29] = 31 qm31 = 124 m31
    // PartialEcMul: LookupElementsBasic<73> - z + alpha + alpha_powers[73] = 75 qm31 = 300 m31
    // RangeCheck_5_4: LookupElementsBasic<2> - 4 qm31 = 16 m31
    // RangeCheck_8: LookupElementsBasic<1> - z + alpha + alpha_powers[1] = 3 qm31 = 12 m31

    MemoryAddressToId* d_mem_addr2id_rel;
    MemoryIdToBig* d_mem_id2big_rel;
    PartialEcMul* d_partial_ec_mul_rel;
    RangeCheck_5_4* d_rc54_rel;
    RangeCheck_8* d_rc8_rel;

    d_mem_addr2id_rel = cuda_malloc<MemoryAddressToId>(1);
    d_mem_id2big_rel = cuda_malloc<MemoryIdToBig>(1);
    d_partial_ec_mul_rel = cuda_malloc<PartialEcMul>(1);
    d_rc54_rel = cuda_malloc<RangeCheck_5_4>(1);
    d_rc8_rel = cuda_malloc<RangeCheck_8>(1);

    cudaMemcpy(d_mem_addr2id_rel, memory_address_to_id_relation, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_id2big_rel, memory_id_to_big_relation, sizeof(MemoryIdToBig), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partial_ec_mul_rel, partial_ec_mul_relation, sizeof(PartialEcMul), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc54_rel, range_check_5_4_relation, sizeof(RangeCheck_5_4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc8_rel, range_check_8_relation, sizeof(RangeCheck_8), cudaMemcpyHostToDevice);

    // Process each logup column - accumulate fractions without per-column prefix sum
    // Column 0: range_check_5_4_0 + memory_address_to_id_0
    gen_pedersen_interaction_col_gen_add_kernel<2, 2><<<num_blocks, BLOCK_SIZE>>>(
        d_rc54_rel, d_mem_addr2id_rel,
        device_lookup_rc54_0, device_lookup_addr2id_0,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        0, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 1: memory_id_to_big_0 + range_check_5_4_1
    gen_pedersen_interaction_col_gen_add_kernel<29, 2><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_id2big_rel, d_rc54_rel,
        device_lookup_id2big_0, device_lookup_rc54_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        1, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 2: memory_address_to_id_1 + memory_id_to_big_1
    gen_pedersen_interaction_col_gen_add_kernel<2, 29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_addr2id_rel, d_mem_id2big_rel,
        device_lookup_addr2id_1, device_lookup_id2big_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        2, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 3: range_check_8_0 + range_check_8_1
    gen_pedersen_interaction_col_gen_add_kernel<1, 1><<<num_blocks, BLOCK_SIZE>>>(
        d_rc8_rel, d_rc8_rel,
        device_lookup_rc8_0, device_lookup_rc8_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        3, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 4: range_check_8_2 + range_check_8_3
    gen_pedersen_interaction_col_gen_add_kernel<1, 1><<<num_blocks, BLOCK_SIZE>>>(
        d_rc8_rel, d_rc8_rel,
        device_lookup_rc8_2, device_lookup_rc8_3,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        4, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 5: partial_ec_mul_0 - partial_ec_mul_1 (SUBTRACTION!)
    gen_pedersen_interaction_col_gen_sub_kernel<73, 73><<<num_blocks, BLOCK_SIZE>>>(
        d_partial_ec_mul_rel, d_partial_ec_mul_rel,
        device_lookup_ec_mul_0, device_lookup_ec_mul_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        5, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 6: partial_ec_mul_2 - partial_ec_mul_3 (SUBTRACTION!)
    gen_pedersen_interaction_col_gen_sub_kernel<73, 73><<<num_blocks, BLOCK_SIZE>>>(
        d_partial_ec_mul_rel, d_partial_ec_mul_rel,
        device_lookup_ec_mul_2, device_lookup_ec_mul_3,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        6, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 7: partial_ec_mul_4 - partial_ec_mul_5 (SUBTRACTION!)
    gen_pedersen_interaction_col_gen_sub_kernel<73, 73><<<num_blocks, BLOCK_SIZE>>>(
        d_partial_ec_mul_rel, d_partial_ec_mul_rel,
        device_lookup_ec_mul_4, device_lookup_ec_mul_5,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        7, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 8: partial_ec_mul_6 - partial_ec_mul_7 (SUBTRACTION!)
    gen_pedersen_interaction_col_gen_sub_kernel<73, 73><<<num_blocks, BLOCK_SIZE>>>(
        d_partial_ec_mul_rel, d_partial_ec_mul_rel,
        device_lookup_ec_mul_6, device_lookup_ec_mul_7,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        8, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Column 9: memory_address_to_id_2 + memory_id_to_big_2
    gen_pedersen_interaction_col_gen_add_kernel<2, 29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_addr2id_rel, d_mem_id2big_rel,
        device_lookup_addr2id_2, device_lookup_id2big_2,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_pedersen_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        9, n_rows, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());


    // After all columns: compute cumsum_shift, apply coord_prefix_sum, and inclusive_prefix_sum on last column
    // Step 1: Compute cumsum (sum of all elements in last column) -> claimed_sum
    // Zero out claimed_sum first since cumsum_shift uses atomic_add
    cudaMemset(claimed_sum, 0, 4 * sizeof(m31));
    size_t shared_size = 4 * BLOCK_SIZE * sizeof(m31);
    gen_pedersen_interaction_cumsum_shift_kernel<<<num_blocks, BLOCK_SIZE, shared_size>>>(
        N_LOGUP_COLS, n_rows, device_interaction_trace, claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 2: Subtract cumsum_shift from last column (coord_prefix_sum)
    gen_pedersen_interaction_coord_prefix_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(
        claimed_sum, N_LOGUP_COLS, n_rows, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("generate pedersen_builtin interaction trace");

    // Step 3: Apply inclusive prefix sum ONLY on the last QM31 column (columns 36-39)
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 4], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 3], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 2], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 1], n_rows);

    // Cleanup
    cuda_free_memory(d_denom);
    cuda_free_memory(d_denom_inv);
    cuda_free_memory(d_numerator0);
    cuda_free_memory(d_numerator1);
    cuda_free_memory(d_numerator2);
    cuda_free_memory(d_numerator3);
    cuda_free_memory(d_mem_addr2id_rel);
    cuda_free_memory(d_mem_id2big_rel);
    cuda_free_memory(d_partial_ec_mul_rel);
    cuda_free_memory(d_rc54_rel);
    cuda_free_memory(d_rc8_rel);

    // Free device memory for pointer arrays
    cuda_free_memory(device_interaction_trace);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_ec_mul_0);
    cuda_free_memory(device_lookup_ec_mul_1);
    cuda_free_memory(device_lookup_ec_mul_2);
    cuda_free_memory(device_lookup_ec_mul_3);
    cuda_free_memory(device_lookup_ec_mul_4);
    cuda_free_memory(device_lookup_ec_mul_5);
    cuda_free_memory(device_lookup_ec_mul_6);
    cuda_free_memory(device_lookup_ec_mul_7);
    cuda_free_memory(device_lookup_rc54_0);
    cuda_free_memory(device_lookup_rc54_1);
    cuda_free_memory(device_lookup_rc8_0);
    cuda_free_memory(device_lookup_rc8_1);
    cuda_free_memory(device_lookup_rc8_2);
    cuda_free_memory(device_lookup_rc8_3);

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
