// ============================================================================
// Poseidon Builtin CUDA Evaluator
// ============================================================================
//
// CUDA version of Poseidon builtin AIR constraint evaluator
// Translated from cairo-air/src/components/poseidon_builtin.rs
//
// ## Overview
// Poseidon builtin performs Poseidon hash operations (a cryptographic hash
// function optimized for zero-knowledge proofs). It reads three field elements
// from memory, applies the Poseidon permutation, and writes three output
// elements back to memory.
//
// ## Data Structure
// - 341 trace columns:
//   - Columns 0-28: input_state_0 (ID + 28 limbs)
//   - Columns 29-57: input_state_1 (ID + 28 limbs)
//   - Columns 58-86: input_state_2 (ID + 28 limbs)
//   - Columns 87-283: Poseidon permutation intermediate values (197 columns)
//     - col87-97: First combination (11 columns)
//     - col98-108: Second combination (11 columns)
//     - col109-119: Third combination (11 columns)
//     - col120-149: First poseidon_full_round_chain_output (30 columns)
//     - col150-159: First cube_252_output (10 columns)
//     - col160-170: Fourth combination (11 columns)
//     - col171-180: Second cube_252_output (10 columns)
//     - col181-191: Fifth combination (11 columns)
//     - col192-231: poseidon_3_partial_rounds_chain_output (40 columns)
//     - col232-242: Sixth combination (11 columns)
//     - col243-253: Seventh combination (11 columns)
//     - col254-283: Second poseidon_full_round_chain_output (30 columns)
//   - Columns 284-301: output_state_0 unpacked limbs (18 unpacked)
//   - Column 302: output_state_0_id
//   - Columns 303-320: output_state_1 unpacked limbs (18 unpacked)
//   - Columns 321-339: output_state_2 unpacked limbs (18 unpacked)
//   - Column 340: output_state_2_id
//
// ## Constraint Logic
// 1. ReadPositiveNumBits252: Read 3 input states from memory (3 uses)
// 2. PoseidonHadesPermutation: Apply Poseidon Hades permutation
//    - Performs full rounds, partial rounds, and state transformations
//    - Uses 197 intermediate trace columns
//    - Includes multiple relation lookups for verification
// 3. Felt252UnpackFrom27 + MemVerify: Write 3 output states to memory (3 uses)
//
// ## Relation Lookups
// - MemoryAddressToId: 6 uses (3 reads + 3 writes)
// - MemoryIdToBig: 6 uses (3 reads + 3 writes)
// - PoseidonFullRoundChain: 2 uses
// - RangeCheckFelt252Width27: 2 uses
// - Cube252: 2 uses
// - RangeCheck_3_3_3_3_3: 2 uses
// - RangeCheck_4_4_4_4: 6 uses
// - RangeCheck_4_4: 3 uses
// - Poseidon3PartialRoundsChain: 1 use
//
// ============================================================================

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_poseidon_builtin.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_felt_252_unpack_from_27.cuh"
#include "evaluate_mem_verify.cuh"
#include "evaluate_common.cuh"
#include "constraints/evaluate_poseidon_hades_permutation.cuh"

#define POSEIDON_BUILTIN_THREAD_COUNT_MAX 256

// ============================================================================
// Poseidon Pre-Kernel: Main constraint evaluation
// ============================================================================
// This kernel reads all 341 trace columns and evaluates constraints
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonBuiltin_Eval *poseidon_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) {
        return;
    }

    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    EvaluatorT cuda_evaluator1(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // ===================== Constants =====================
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_3 = m31(3);
    const m31 M31_4 = m31(4);
    const m31 M31_5 = m31(5);
    const m31 M31_6 = m31(6);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);

    // ===================== Read preprocessed column (Seq) =====================
    m31 seq = cuda_evaluator0.next_trace_mask();

    // ===================== Read all 341 trace columns =====================
    // Input state 0: columns 0-28 (ID + 28 limbs)
    m31 input_state_0_id_col0 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_0_col1 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_1_col2 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_2_col3 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_3_col4 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_4_col5 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_5_col6 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_6_col7 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_7_col8 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_8_col9 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_9_col10 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_10_col11 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_11_col12 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_12_col13 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_13_col14 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_14_col15 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_15_col16 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_16_col17 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_17_col18 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_18_col19 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_19_col20 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_20_col21 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_21_col22 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_22_col23 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_23_col24 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_24_col25 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_25_col26 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_26_col27 = cuda_evaluator1.next_trace_mask();
    m31 input_state_0_limb_27_col28 = cuda_evaluator1.next_trace_mask();

    // Input state 1: columns 29-57
    m31 input_state_1_id_col29 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_0_col30 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_1_col31 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_2_col32 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_3_col33 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_4_col34 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_5_col35 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_6_col36 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_7_col37 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_8_col38 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_9_col39 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_10_col40 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_11_col41 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_12_col42 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_13_col43 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_14_col44 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_15_col45 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_16_col46 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_17_col47 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_18_col48 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_19_col49 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_20_col50 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_21_col51 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_22_col52 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_23_col53 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_24_col54 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_25_col55 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_26_col56 = cuda_evaluator1.next_trace_mask();
    m31 input_state_1_limb_27_col57 = cuda_evaluator1.next_trace_mask();

    // Input state 2: columns 58-86
    m31 input_state_2_id_col58 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_0_col59 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_1_col60 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_2_col61 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_3_col62 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_4_col63 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_5_col64 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_6_col65 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_7_col66 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_8_col67 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_9_col68 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_10_col69 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_11_col70 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_12_col71 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_13_col72 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_14_col73 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_15_col74 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_16_col75 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_17_col76 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_18_col77 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_19_col78 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_20_col79 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_21_col80 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_22_col81 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_23_col82 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_24_col83 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_25_col84 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_26_col85 = cuda_evaluator1.next_trace_mask();
    m31 input_state_2_limb_27_col86 = cuda_evaluator1.next_trace_mask();

    // Poseidon permutation intermediate columns: 87-283 (197 columns)
    // These are used in the complex Poseidon Hades permutation logic
    // First combination (col87-97): 11 columns
    m31 combination_limb_0_col87 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col88 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col89 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col90 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col91 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col92 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col93 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col94 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col95 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col96 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col97 = cuda_evaluator1.next_trace_mask();

    // Second combination (col98-108): 11 columns
    m31 combination_limb_0_col98 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col99 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col100 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col101 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col102 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col103 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col104 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col105 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col106 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col107 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col108 = cuda_evaluator1.next_trace_mask();

    // Third combination (col109-119): 11 columns
    m31 combination_limb_0_col109 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col110 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col111 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col112 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col113 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col114 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col115 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col116 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col117 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col118 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col119 = cuda_evaluator1.next_trace_mask();

    // First poseidon_full_round_chain_output (col120-149): 30 columns
    m31 poseidon_full_round_chain_output_limb_0_col120 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_1_col121 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_2_col122 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_3_col123 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_4_col124 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_5_col125 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_6_col126 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_7_col127 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_8_col128 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_9_col129 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_10_col130 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_11_col131 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_12_col132 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_13_col133 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_14_col134 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_15_col135 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_16_col136 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_17_col137 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_18_col138 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_19_col139 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_20_col140 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_21_col141 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_22_col142 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_23_col143 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_24_col144 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_25_col145 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_26_col146 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_27_col147 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_28_col148 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_29_col149 = cuda_evaluator1.next_trace_mask();

    // First cube_252_output (col150-159): 10 columns
    m31 cube_252_output_limb_0_col150 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_1_col151 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_2_col152 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_3_col153 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_4_col154 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_5_col155 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_6_col156 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_7_col157 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_8_col158 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_9_col159 = cuda_evaluator1.next_trace_mask();

    // Fourth combination (col160-170): 11 columns
    m31 combination_limb_0_col160 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col161 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col162 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col163 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col164 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col165 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col166 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col167 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col168 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col169 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col170 = cuda_evaluator1.next_trace_mask();

    // Second cube_252_output (col171-180): 10 columns
    m31 cube_252_output_limb_0_col171 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_1_col172 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_2_col173 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_3_col174 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_4_col175 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_5_col176 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_6_col177 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_7_col178 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_8_col179 = cuda_evaluator1.next_trace_mask();
    m31 cube_252_output_limb_9_col180 = cuda_evaluator1.next_trace_mask();

    // Fifth combination (col181-191): 11 columns
    m31 combination_limb_0_col181 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col182 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col183 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col184 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col185 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col186 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col187 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col188 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col189 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col190 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col191 = cuda_evaluator1.next_trace_mask();

    // poseidon_3_partial_rounds_chain_output (col192-231): 40 columns
    m31 poseidon_3_partial_rounds_chain_output_limb_0_col192 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_1_col193 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_2_col194 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_3_col195 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_4_col196 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_5_col197 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_6_col198 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_7_col199 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_8_col200 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_9_col201 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_10_col202 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_11_col203 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_12_col204 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_13_col205 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_14_col206 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_15_col207 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_16_col208 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_17_col209 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_18_col210 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_19_col211 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_20_col212 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_21_col213 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_22_col214 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_23_col215 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_24_col216 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_25_col217 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_26_col218 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_27_col219 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_28_col220 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_29_col221 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_30_col222 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_31_col223 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_32_col224 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_33_col225 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_34_col226 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_35_col227 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_36_col228 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_37_col229 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_38_col230 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_39_col231 = cuda_evaluator1.next_trace_mask();

    // Sixth combination (col232-242): 11 columns
    m31 combination_limb_0_col232 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col233 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col234 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col235 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col236 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col237 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col238 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col239 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col240 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col241 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col242 = cuda_evaluator1.next_trace_mask();

    // Seventh combination (col243-253): 11 columns
    m31 combination_limb_0_col243 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_1_col244 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_2_col245 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_3_col246 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_4_col247 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_5_col248 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_6_col249 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_7_col250 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_8_col251 = cuda_evaluator1.next_trace_mask();
    m31 combination_limb_9_col252 = cuda_evaluator1.next_trace_mask();
    m31 p_coef_col253 = cuda_evaluator1.next_trace_mask();

    // Second poseidon_full_round_chain_output (col254-283): 30 columns
    // These are the outputs from PoseidonHadesPermutation
    m31 poseidon_full_round_chain_output_limb_0_col254 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_1_col255 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_2_col256 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_3_col257 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_4_col258 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_5_col259 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_6_col260 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_7_col261 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_8_col262 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_9_col263 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_10_col264 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_11_col265 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_12_col266 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_13_col267 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_14_col268 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_15_col269 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_16_col270 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_17_col271 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_18_col272 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_19_col273 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_20_col274 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_21_col275 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_22_col276 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_23_col277 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_24_col278 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_25_col279 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_26_col280 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_27_col281 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_28_col282 = cuda_evaluator1.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_29_col283 = cuda_evaluator1.next_trace_mask();

    // Unpacked limbs for output state 0: columns 284-301
    m31 unpacked_limb_0_col284 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_1_col285 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_3_col286 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_4_col287 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_6_col288 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_7_col289 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_9_col290 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_10_col291 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_12_col292 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_13_col293 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_15_col294 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_16_col295 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_18_col296 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_19_col297 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_21_col298 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_22_col299 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_24_col300 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_25_col301 = cuda_evaluator1.next_trace_mask();

    m31 output_state_0_id_col302 = cuda_evaluator1.next_trace_mask();

    // Unpacked limbs for output state 1: columns 303-320
    m31 unpacked_limb_0_col303 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_1_col304 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_3_col305 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_4_col306 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_6_col307 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_7_col308 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_9_col309 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_10_col310 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_12_col311 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_13_col312 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_15_col313 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_16_col314 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_18_col315 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_19_col316 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_21_col317 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_22_col318 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_24_col319 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_25_col320 = cuda_evaluator1.next_trace_mask();

    m31 output_state_1_id_col321 = cuda_evaluator1.next_trace_mask();

    // Unpacked limbs for output state 2: columns 322-339
    m31 unpacked_limb_0_col322 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_1_col323 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_3_col324 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_4_col325 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_6_col326 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_7_col327 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_9_col328 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_10_col329 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_12_col330 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_13_col331 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_15_col332 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_16_col333 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_18_col334 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_19_col335 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_21_col336 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_22_col337 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_24_col338 = cuda_evaluator1.next_trace_mask();
    m31 unpacked_limb_25_col339 = cuda_evaluator1.next_trace_mask();

    m31 output_state_2_id_col340 = cuda_evaluator1.next_trace_mask();

    // ===================== Compute instance address =====================
    // instance_addr = seq * 6 + poseidon_builtin_segment_start
    m31 instance_addr = add(
        mul(seq, M31_6),
        m31(poseidon_eval->claim.poseidon_builtin_segment_start)
    );

    // ===================== ReadPositiveNumBits252: Input state 0 =====================
    m31 output_vec_0[29];
    evaluate_read_positive_num_bits_252<EvaluatorT>(
        instance_addr,
        input_state_0_id_col0,
        input_state_0_limb_0_col1, input_state_0_limb_1_col2, input_state_0_limb_2_col3,
        input_state_0_limb_3_col4, input_state_0_limb_4_col5, input_state_0_limb_5_col6,
        input_state_0_limb_6_col7, input_state_0_limb_7_col8, input_state_0_limb_8_col9,
        input_state_0_limb_9_col10, input_state_0_limb_10_col11, input_state_0_limb_11_col12,
        input_state_0_limb_12_col13, input_state_0_limb_13_col14, input_state_0_limb_14_col15,
        input_state_0_limb_15_col16, input_state_0_limb_16_col17, input_state_0_limb_17_col18,
        input_state_0_limb_18_col19, input_state_0_limb_19_col20, input_state_0_limb_20_col21,
        input_state_0_limb_21_col22, input_state_0_limb_22_col23, input_state_0_limb_23_col24,
        input_state_0_limb_24_col25, input_state_0_limb_25_col26, input_state_0_limb_26_col27,
        input_state_0_limb_27_col28,
        output_vec_0,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== ReadPositiveNumBits252: Input state 1 =====================
    m31 output_vec_1[29];
    evaluate_read_positive_num_bits_252<EvaluatorT>(
        add(instance_addr, M31_1),
        input_state_1_id_col29,
        input_state_1_limb_0_col30, input_state_1_limb_1_col31, input_state_1_limb_2_col32,
        input_state_1_limb_3_col33, input_state_1_limb_4_col34, input_state_1_limb_5_col35,
        input_state_1_limb_6_col36, input_state_1_limb_7_col37, input_state_1_limb_8_col38,
        input_state_1_limb_9_col39, input_state_1_limb_10_col40, input_state_1_limb_11_col41,
        input_state_1_limb_12_col42, input_state_1_limb_13_col43, input_state_1_limb_14_col44,
        input_state_1_limb_15_col45, input_state_1_limb_16_col46, input_state_1_limb_17_col47,
        input_state_1_limb_18_col48, input_state_1_limb_19_col49, input_state_1_limb_20_col50,
        input_state_1_limb_21_col51, input_state_1_limb_22_col52, input_state_1_limb_23_col53,
        input_state_1_limb_24_col54, input_state_1_limb_25_col55, input_state_1_limb_26_col56,
        input_state_1_limb_27_col57,
        output_vec_1,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== ReadPositiveNumBits252: Input state 2 =====================
    m31 output_vec_2[29];
    evaluate_read_positive_num_bits_252<EvaluatorT>(
        add(instance_addr, M31_2),
        input_state_2_id_col58,
        input_state_2_limb_0_col59, input_state_2_limb_1_col60, input_state_2_limb_2_col61,
        input_state_2_limb_3_col62, input_state_2_limb_4_col63, input_state_2_limb_5_col64,
        input_state_2_limb_6_col65, input_state_2_limb_7_col66, input_state_2_limb_8_col67,
        input_state_2_limb_9_col68, input_state_2_limb_10_col69, input_state_2_limb_11_col70,
        input_state_2_limb_12_col71, input_state_2_limb_13_col72, input_state_2_limb_14_col73,
        input_state_2_limb_15_col74, input_state_2_limb_16_col75, input_state_2_limb_17_col76,
        input_state_2_limb_18_col77, input_state_2_limb_19_col78, input_state_2_limb_20_col79,
        input_state_2_limb_21_col80, input_state_2_limb_22_col81, input_state_2_limb_23_col82,
        input_state_2_limb_24_col83, input_state_2_limb_25_col84, input_state_2_limb_26_col85,
        input_state_2_limb_27_col86,
        output_vec_2,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== Pack Input Limbs for PoseidonHadesPermutation =====================
    // ReadPositiveNumBits252 reads 28 9-bit limbs from memory.
    // PoseidonHadesPermutation expects 10 27-bit packed limbs per state.
    // We need to pack every 3 consecutive 9-bit limbs into one 27-bit limb:
    //   packed_limb_i = limb_{3i} + limb_{3i+1} * 512 + limb_{3i+2} * 262144
    // The 10th limb (limb_27) is used directly without packing.

    // Pack input state 0 (from input_state_0_limb_0 to input_state_0_limb_26)
    m31 packed_input_state_0_limb_0 = add(add(
        input_state_0_limb_0_col1,
        mul(input_state_0_limb_1_col2, M31_512)),
        mul(input_state_0_limb_2_col3, M31_262144));
    m31 packed_input_state_0_limb_1 = add(add(
        input_state_0_limb_3_col4,
        mul(input_state_0_limb_4_col5, M31_512)),
        mul(input_state_0_limb_5_col6, M31_262144));
    m31 packed_input_state_0_limb_2 = add(add(
        input_state_0_limb_6_col7,
        mul(input_state_0_limb_7_col8, M31_512)),
        mul(input_state_0_limb_8_col9, M31_262144));
    m31 packed_input_state_0_limb_3 = add(add(
        input_state_0_limb_9_col10,
        mul(input_state_0_limb_10_col11, M31_512)),
        mul(input_state_0_limb_11_col12, M31_262144));
    m31 packed_input_state_0_limb_4 = add(add(
        input_state_0_limb_12_col13,
        mul(input_state_0_limb_13_col14, M31_512)),
        mul(input_state_0_limb_14_col15, M31_262144));
    m31 packed_input_state_0_limb_5 = add(add(
        input_state_0_limb_15_col16,
        mul(input_state_0_limb_16_col17, M31_512)),
        mul(input_state_0_limb_17_col18, M31_262144));
    m31 packed_input_state_0_limb_6 = add(add(
        input_state_0_limb_18_col19,
        mul(input_state_0_limb_19_col20, M31_512)),
        mul(input_state_0_limb_20_col21, M31_262144));
    m31 packed_input_state_0_limb_7 = add(add(
        input_state_0_limb_21_col22,
        mul(input_state_0_limb_22_col23, M31_512)),
        mul(input_state_0_limb_23_col24, M31_262144));
    m31 packed_input_state_0_limb_8 = add(add(
        input_state_0_limb_24_col25,
        mul(input_state_0_limb_25_col26, M31_512)),
        mul(input_state_0_limb_26_col27, M31_262144));
    // limb_27 is used directly
    m31 packed_input_state_0_limb_9 = input_state_0_limb_27_col28;

    // Pack input state 1 (from input_state_1_limb_0 to input_state_1_limb_26)
    m31 packed_input_state_1_limb_0 = add(add(
        input_state_1_limb_0_col30,
        mul(input_state_1_limb_1_col31, M31_512)),
        mul(input_state_1_limb_2_col32, M31_262144));
    m31 packed_input_state_1_limb_1 = add(add(
        input_state_1_limb_3_col33,
        mul(input_state_1_limb_4_col34, M31_512)),
        mul(input_state_1_limb_5_col35, M31_262144));
    m31 packed_input_state_1_limb_2 = add(add(
        input_state_1_limb_6_col36,
        mul(input_state_1_limb_7_col37, M31_512)),
        mul(input_state_1_limb_8_col38, M31_262144));
    m31 packed_input_state_1_limb_3 = add(add(
        input_state_1_limb_9_col39,
        mul(input_state_1_limb_10_col40, M31_512)),
        mul(input_state_1_limb_11_col41, M31_262144));
    m31 packed_input_state_1_limb_4 = add(add(
        input_state_1_limb_12_col42,
        mul(input_state_1_limb_13_col43, M31_512)),
        mul(input_state_1_limb_14_col44, M31_262144));
    m31 packed_input_state_1_limb_5 = add(add(
        input_state_1_limb_15_col45,
        mul(input_state_1_limb_16_col46, M31_512)),
        mul(input_state_1_limb_17_col47, M31_262144));
    m31 packed_input_state_1_limb_6 = add(add(
        input_state_1_limb_18_col48,
        mul(input_state_1_limb_19_col49, M31_512)),
        mul(input_state_1_limb_20_col50, M31_262144));
    m31 packed_input_state_1_limb_7 = add(add(
        input_state_1_limb_21_col51,
        mul(input_state_1_limb_22_col52, M31_512)),
        mul(input_state_1_limb_23_col53, M31_262144));
    m31 packed_input_state_1_limb_8 = add(add(
        input_state_1_limb_24_col54,
        mul(input_state_1_limb_25_col55, M31_512)),
        mul(input_state_1_limb_26_col56, M31_262144));
    // limb_27 is used directly
    m31 packed_input_state_1_limb_9 = input_state_1_limb_27_col57;

    // Pack input state 2 (from input_state_2_limb_0 to input_state_2_limb_26)
    m31 packed_input_state_2_limb_0 = add(add(
        input_state_2_limb_0_col59,
        mul(input_state_2_limb_1_col60, M31_512)),
        mul(input_state_2_limb_2_col61, M31_262144));
    m31 packed_input_state_2_limb_1 = add(add(
        input_state_2_limb_3_col62,
        mul(input_state_2_limb_4_col63, M31_512)),
        mul(input_state_2_limb_5_col64, M31_262144));
    m31 packed_input_state_2_limb_2 = add(add(
        input_state_2_limb_6_col65,
        mul(input_state_2_limb_7_col66, M31_512)),
        mul(input_state_2_limb_8_col67, M31_262144));
    m31 packed_input_state_2_limb_3 = add(add(
        input_state_2_limb_9_col68,
        mul(input_state_2_limb_10_col69, M31_512)),
        mul(input_state_2_limb_11_col70, M31_262144));
    m31 packed_input_state_2_limb_4 = add(add(
        input_state_2_limb_12_col71,
        mul(input_state_2_limb_13_col72, M31_512)),
        mul(input_state_2_limb_14_col73, M31_262144));
    m31 packed_input_state_2_limb_5 = add(add(
        input_state_2_limb_15_col74,
        mul(input_state_2_limb_16_col75, M31_512)),
        mul(input_state_2_limb_17_col76, M31_262144));
    m31 packed_input_state_2_limb_6 = add(add(
        input_state_2_limb_18_col77,
        mul(input_state_2_limb_19_col78, M31_512)),
        mul(input_state_2_limb_20_col79, M31_262144));
    m31 packed_input_state_2_limb_7 = add(add(
        input_state_2_limb_21_col80,
        mul(input_state_2_limb_22_col81, M31_512)),
        mul(input_state_2_limb_23_col82, M31_262144));
    m31 packed_input_state_2_limb_8 = add(add(
        input_state_2_limb_24_col83,
        mul(input_state_2_limb_25_col84, M31_512)),
        mul(input_state_2_limb_26_col85, M31_262144));
    // limb_27 is used directly
    m31 packed_input_state_2_limb_9 = input_state_2_limb_27_col86;

    // ===================== PoseidonHadesPermutation =====================
    // The Poseidon Hades permutation is the core operation of the Poseidon hash.
    // It takes 30 packed input limbs (27-bit limbs) and produces the final output.
    // Each state contributes 10 packed limbs (9 packed + 1 direct).

    poseidon_hades_permutation_evaluate<EvaluatorT>(
        // Input limbs (30 total - 3 states of 10 limbs each)
        // Packed input state 0 (10 limbs)
        packed_input_state_0_limb_0, packed_input_state_0_limb_1, packed_input_state_0_limb_2,
        packed_input_state_0_limb_3, packed_input_state_0_limb_4, packed_input_state_0_limb_5,
        packed_input_state_0_limb_6, packed_input_state_0_limb_7, packed_input_state_0_limb_8,
        packed_input_state_0_limb_9,
        // Packed input state 1 (10 limbs)
        packed_input_state_1_limb_0, packed_input_state_1_limb_1, packed_input_state_1_limb_2,
        packed_input_state_1_limb_3, packed_input_state_1_limb_4, packed_input_state_1_limb_5,
        packed_input_state_1_limb_6, packed_input_state_1_limb_7, packed_input_state_1_limb_8,
        packed_input_state_1_limb_9,
        // Packed input state 2 (10 limbs)
        packed_input_state_2_limb_0, packed_input_state_2_limb_1, packed_input_state_2_limb_2,
        packed_input_state_2_limb_3, packed_input_state_2_limb_4, packed_input_state_2_limb_5,
        packed_input_state_2_limb_6, packed_input_state_2_limb_7, packed_input_state_2_limb_8,
        packed_input_state_2_limb_9,

        // First combination (col87-97): 11 columns
        combination_limb_0_col87, combination_limb_1_col88, combination_limb_2_col89,
        combination_limb_3_col90, combination_limb_4_col91, combination_limb_5_col92,
        combination_limb_6_col93, combination_limb_7_col94, combination_limb_8_col95,
        combination_limb_9_col96, p_coef_col97,

        // Second combination (col98-108): 11 columns
        combination_limb_0_col98, combination_limb_1_col99, combination_limb_2_col100,
        combination_limb_3_col101, combination_limb_4_col102, combination_limb_5_col103,
        combination_limb_6_col104, combination_limb_7_col105, combination_limb_8_col106,
        combination_limb_9_col107, p_coef_col108,

        // Third combination (col109-119): 11 columns
        combination_limb_0_col109, combination_limb_1_col110, combination_limb_2_col111,
        combination_limb_3_col112, combination_limb_4_col113, combination_limb_5_col114,
        combination_limb_6_col115, combination_limb_7_col116, combination_limb_8_col117,
        combination_limb_9_col118, p_coef_col119,

        // First poseidon_full_round_chain_output (col120-149): 30 columns
        poseidon_full_round_chain_output_limb_0_col120, poseidon_full_round_chain_output_limb_1_col121,
        poseidon_full_round_chain_output_limb_2_col122, poseidon_full_round_chain_output_limb_3_col123,
        poseidon_full_round_chain_output_limb_4_col124, poseidon_full_round_chain_output_limb_5_col125,
        poseidon_full_round_chain_output_limb_6_col126, poseidon_full_round_chain_output_limb_7_col127,
        poseidon_full_round_chain_output_limb_8_col128, poseidon_full_round_chain_output_limb_9_col129,
        poseidon_full_round_chain_output_limb_10_col130, poseidon_full_round_chain_output_limb_11_col131,
        poseidon_full_round_chain_output_limb_12_col132, poseidon_full_round_chain_output_limb_13_col133,
        poseidon_full_round_chain_output_limb_14_col134, poseidon_full_round_chain_output_limb_15_col135,
        poseidon_full_round_chain_output_limb_16_col136, poseidon_full_round_chain_output_limb_17_col137,
        poseidon_full_round_chain_output_limb_18_col138, poseidon_full_round_chain_output_limb_19_col139,
        poseidon_full_round_chain_output_limb_20_col140, poseidon_full_round_chain_output_limb_21_col141,
        poseidon_full_round_chain_output_limb_22_col142, poseidon_full_round_chain_output_limb_23_col143,
        poseidon_full_round_chain_output_limb_24_col144, poseidon_full_round_chain_output_limb_25_col145,
        poseidon_full_round_chain_output_limb_26_col146, poseidon_full_round_chain_output_limb_27_col147,
        poseidon_full_round_chain_output_limb_28_col148, poseidon_full_round_chain_output_limb_29_col149,

        // First cube_252_output (col150-159): 10 columns
        cube_252_output_limb_0_col150, cube_252_output_limb_1_col151,
        cube_252_output_limb_2_col152, cube_252_output_limb_3_col153,
        cube_252_output_limb_4_col154, cube_252_output_limb_5_col155,
        cube_252_output_limb_6_col156, cube_252_output_limb_7_col157,
        cube_252_output_limb_8_col158, cube_252_output_limb_9_col159,

        // Fourth combination (col160-170): 11 columns
        combination_limb_0_col160, combination_limb_1_col161, combination_limb_2_col162,
        combination_limb_3_col163, combination_limb_4_col164, combination_limb_5_col165,
        combination_limb_6_col166, combination_limb_7_col167, combination_limb_8_col168,
        combination_limb_9_col169, p_coef_col170,

        // Second cube_252_output (col171-180): 10 columns
        cube_252_output_limb_0_col171, cube_252_output_limb_1_col172,
        cube_252_output_limb_2_col173, cube_252_output_limb_3_col174,
        cube_252_output_limb_4_col175, cube_252_output_limb_5_col176,
        cube_252_output_limb_6_col177, cube_252_output_limb_7_col178,
        cube_252_output_limb_8_col179, cube_252_output_limb_9_col180,

        // Fifth combination (col181-191): 11 columns
        combination_limb_0_col181, combination_limb_1_col182, combination_limb_2_col183,
        combination_limb_3_col184, combination_limb_4_col185, combination_limb_5_col186,
        combination_limb_6_col187, combination_limb_7_col188, combination_limb_8_col189,
        combination_limb_9_col190, p_coef_col191,

        // poseidon_3_partial_rounds_chain_output (col192-231): 40 columns
        poseidon_3_partial_rounds_chain_output_limb_0_col192, poseidon_3_partial_rounds_chain_output_limb_1_col193,
        poseidon_3_partial_rounds_chain_output_limb_2_col194, poseidon_3_partial_rounds_chain_output_limb_3_col195,
        poseidon_3_partial_rounds_chain_output_limb_4_col196, poseidon_3_partial_rounds_chain_output_limb_5_col197,
        poseidon_3_partial_rounds_chain_output_limb_6_col198, poseidon_3_partial_rounds_chain_output_limb_7_col199,
        poseidon_3_partial_rounds_chain_output_limb_8_col200, poseidon_3_partial_rounds_chain_output_limb_9_col201,
        poseidon_3_partial_rounds_chain_output_limb_10_col202, poseidon_3_partial_rounds_chain_output_limb_11_col203,
        poseidon_3_partial_rounds_chain_output_limb_12_col204, poseidon_3_partial_rounds_chain_output_limb_13_col205,
        poseidon_3_partial_rounds_chain_output_limb_14_col206, poseidon_3_partial_rounds_chain_output_limb_15_col207,
        poseidon_3_partial_rounds_chain_output_limb_16_col208, poseidon_3_partial_rounds_chain_output_limb_17_col209,
        poseidon_3_partial_rounds_chain_output_limb_18_col210, poseidon_3_partial_rounds_chain_output_limb_19_col211,
        poseidon_3_partial_rounds_chain_output_limb_20_col212, poseidon_3_partial_rounds_chain_output_limb_21_col213,
        poseidon_3_partial_rounds_chain_output_limb_22_col214, poseidon_3_partial_rounds_chain_output_limb_23_col215,
        poseidon_3_partial_rounds_chain_output_limb_24_col216, poseidon_3_partial_rounds_chain_output_limb_25_col217,
        poseidon_3_partial_rounds_chain_output_limb_26_col218, poseidon_3_partial_rounds_chain_output_limb_27_col219,
        poseidon_3_partial_rounds_chain_output_limb_28_col220, poseidon_3_partial_rounds_chain_output_limb_29_col221,
        poseidon_3_partial_rounds_chain_output_limb_30_col222, poseidon_3_partial_rounds_chain_output_limb_31_col223,
        poseidon_3_partial_rounds_chain_output_limb_32_col224, poseidon_3_partial_rounds_chain_output_limb_33_col225,
        poseidon_3_partial_rounds_chain_output_limb_34_col226, poseidon_3_partial_rounds_chain_output_limb_35_col227,
        poseidon_3_partial_rounds_chain_output_limb_36_col228, poseidon_3_partial_rounds_chain_output_limb_37_col229,
        poseidon_3_partial_rounds_chain_output_limb_38_col230, poseidon_3_partial_rounds_chain_output_limb_39_col231,

        // Sixth combination (col232-242): 11 columns
        combination_limb_0_col232, combination_limb_1_col233, combination_limb_2_col234,
        combination_limb_3_col235, combination_limb_4_col236, combination_limb_5_col237,
        combination_limb_6_col238, combination_limb_7_col239, combination_limb_8_col240,
        combination_limb_9_col241, p_coef_col242,

        // Seventh combination (col243-253): 11 columns
        combination_limb_0_col243, combination_limb_1_col244, combination_limb_2_col245,
        combination_limb_3_col246, combination_limb_4_col247, combination_limb_5_col248,
        combination_limb_6_col249, combination_limb_7_col250, combination_limb_8_col251,
        combination_limb_9_col252, p_coef_col253,

        // Second poseidon_full_round_chain_output (col254-283): 30 columns
        poseidon_full_round_chain_output_limb_0_col254, poseidon_full_round_chain_output_limb_1_col255,
        poseidon_full_round_chain_output_limb_2_col256, poseidon_full_round_chain_output_limb_3_col257,
        poseidon_full_round_chain_output_limb_4_col258, poseidon_full_round_chain_output_limb_5_col259,
        poseidon_full_round_chain_output_limb_6_col260, poseidon_full_round_chain_output_limb_7_col261,
        poseidon_full_round_chain_output_limb_8_col262, poseidon_full_round_chain_output_limb_9_col263,
        poseidon_full_round_chain_output_limb_10_col264, poseidon_full_round_chain_output_limb_11_col265,
        poseidon_full_round_chain_output_limb_12_col266, poseidon_full_round_chain_output_limb_13_col267,
        poseidon_full_round_chain_output_limb_14_col268, poseidon_full_round_chain_output_limb_15_col269,
        poseidon_full_round_chain_output_limb_16_col270, poseidon_full_round_chain_output_limb_17_col271,
        poseidon_full_round_chain_output_limb_18_col272, poseidon_full_round_chain_output_limb_19_col273,
        poseidon_full_round_chain_output_limb_20_col274, poseidon_full_round_chain_output_limb_21_col275,
        poseidon_full_round_chain_output_limb_22_col276, poseidon_full_round_chain_output_limb_23_col277,
        poseidon_full_round_chain_output_limb_24_col278, poseidon_full_round_chain_output_limb_25_col279,
        poseidon_full_round_chain_output_limb_26_col280, poseidon_full_round_chain_output_limb_27_col281,
        poseidon_full_round_chain_output_limb_28_col282, poseidon_full_round_chain_output_limb_29_col283,

        // Lookup elements
        poseidon_eval->poseidon_full_round_chain_lookup_elements,
        poseidon_eval->range_check_felt_252_width_27_lookup_elements,
        poseidon_eval->cube_252_lookup_elements,
        poseidon_eval->range_check_3_3_3_3_3_lookup_elements,
        poseidon_eval->range_check_4_4_4_4_lookup_elements,
        poseidon_eval->range_check_4_4_lookup_elements,
        poseidon_eval->poseidon_3_partial_rounds_chain_lookup_elements,

        seq,

        &cuda_evaluator1
    );

    // ===================== Felt252UnpackFrom27 + MemVerify: Output state 0 =====================
    m31 felt_252_unpack_output_0[10];

    felt_252_unpack_from_27_evaluate<EvaluatorT>(
        poseidon_full_round_chain_output_limb_0_col254,
        poseidon_full_round_chain_output_limb_1_col255,
        poseidon_full_round_chain_output_limb_2_col256,
        poseidon_full_round_chain_output_limb_3_col257,
        poseidon_full_round_chain_output_limb_4_col258,
        poseidon_full_round_chain_output_limb_5_col259,
        poseidon_full_round_chain_output_limb_6_col260,
        poseidon_full_round_chain_output_limb_7_col261,
        poseidon_full_round_chain_output_limb_8_col262,
        poseidon_full_round_chain_output_limb_9_col263,
        unpacked_limb_0_col284,
        unpacked_limb_1_col285,
        unpacked_limb_3_col286,
        unpacked_limb_4_col287,
        unpacked_limb_6_col288,
        unpacked_limb_7_col289,
        unpacked_limb_9_col290,
        unpacked_limb_10_col291,
        unpacked_limb_12_col292,
        unpacked_limb_13_col293,
        unpacked_limb_15_col294,
        unpacked_limb_16_col295,
        unpacked_limb_18_col296,
        unpacked_limb_19_col297,
        unpacked_limb_21_col298,
        unpacked_limb_22_col299,
        unpacked_limb_24_col300,
        unpacked_limb_25_col301,
        felt_252_unpack_output_0,
        &cuda_evaluator1
    );

    mem_verify_evaluate<EvaluatorT>(
        add(instance_addr, M31_3),
        unpacked_limb_0_col284,
        unpacked_limb_1_col285,
        felt_252_unpack_output_0[0],  // limb_2
        unpacked_limb_3_col286,
        unpacked_limb_4_col287,
        felt_252_unpack_output_0[1],  // limb_5
        unpacked_limb_6_col288,
        unpacked_limb_7_col289,
        felt_252_unpack_output_0[2],  // limb_8
        unpacked_limb_9_col290,
        unpacked_limb_10_col291,
        felt_252_unpack_output_0[3],  // limb_11
        unpacked_limb_12_col292,
        unpacked_limb_13_col293,
        felt_252_unpack_output_0[4],  // limb_14
        unpacked_limb_15_col294,
        unpacked_limb_16_col295,
        felt_252_unpack_output_0[5],  // limb_17
        unpacked_limb_18_col296,
        unpacked_limb_19_col297,
        felt_252_unpack_output_0[6],  // limb_20
        unpacked_limb_21_col298,
        unpacked_limb_22_col299,
        felt_252_unpack_output_0[7],  // limb_23
        unpacked_limb_24_col300,
        unpacked_limb_25_col301,
        felt_252_unpack_output_0[8],  // limb_26
        poseidon_full_round_chain_output_limb_9_col263,  // limb_27
        output_state_0_id_col302,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== Felt252UnpackFrom27 + MemVerify: Output state 1 =====================
    m31 felt_252_unpack_output_1[10];
    felt_252_unpack_from_27_evaluate<EvaluatorT>(
        poseidon_full_round_chain_output_limb_10_col264,
        poseidon_full_round_chain_output_limb_11_col265,
        poseidon_full_round_chain_output_limb_12_col266,
        poseidon_full_round_chain_output_limb_13_col267,
        poseidon_full_round_chain_output_limb_14_col268,
        poseidon_full_round_chain_output_limb_15_col269,
        poseidon_full_round_chain_output_limb_16_col270,
        poseidon_full_round_chain_output_limb_17_col271,
        poseidon_full_round_chain_output_limb_18_col272,
        poseidon_full_round_chain_output_limb_19_col273,
        unpacked_limb_0_col303,
        unpacked_limb_1_col304,
        unpacked_limb_3_col305,
        unpacked_limb_4_col306,
        unpacked_limb_6_col307,
        unpacked_limb_7_col308,
        unpacked_limb_9_col309,
        unpacked_limb_10_col310,
        unpacked_limb_12_col311,
        unpacked_limb_13_col312,
        unpacked_limb_15_col313,
        unpacked_limb_16_col314,
        unpacked_limb_18_col315,
        unpacked_limb_19_col316,
        unpacked_limb_21_col317,
        unpacked_limb_22_col318,
        unpacked_limb_24_col319,
        unpacked_limb_25_col320,
        felt_252_unpack_output_1,
        &cuda_evaluator1
    );

    mem_verify_evaluate<EvaluatorT>(
        add(instance_addr, M31_4),
        unpacked_limb_0_col303,
        unpacked_limb_1_col304,
        felt_252_unpack_output_1[0],  // limb_2
        unpacked_limb_3_col305,
        unpacked_limb_4_col306,
        felt_252_unpack_output_1[1],  // limb_5
        unpacked_limb_6_col307,
        unpacked_limb_7_col308,
        felt_252_unpack_output_1[2],  // limb_8
        unpacked_limb_9_col309,
        unpacked_limb_10_col310,
        felt_252_unpack_output_1[3],  // limb_11
        unpacked_limb_12_col311,
        unpacked_limb_13_col312,
        felt_252_unpack_output_1[4],  // limb_14
        unpacked_limb_15_col313,
        unpacked_limb_16_col314,
        felt_252_unpack_output_1[5],  // limb_17
        unpacked_limb_18_col315,
        unpacked_limb_19_col316,
        felt_252_unpack_output_1[6],  // limb_20
        unpacked_limb_21_col317,
        unpacked_limb_22_col318,
        felt_252_unpack_output_1[7],  // limb_23
        unpacked_limb_24_col319,
        unpacked_limb_25_col320,
        felt_252_unpack_output_1[8],  // limb_26
        poseidon_full_round_chain_output_limb_19_col273,  // limb_27
        output_state_1_id_col321,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== Felt252UnpackFrom27 + MemVerify: Output state 2 =====================
    m31 felt_252_unpack_output_2[10];
    felt_252_unpack_from_27_evaluate<EvaluatorT>(
        poseidon_full_round_chain_output_limb_20_col274,
        poseidon_full_round_chain_output_limb_21_col275,
        poseidon_full_round_chain_output_limb_22_col276,
        poseidon_full_round_chain_output_limb_23_col277,
        poseidon_full_round_chain_output_limb_24_col278,
        poseidon_full_round_chain_output_limb_25_col279,
        poseidon_full_round_chain_output_limb_26_col280,
        poseidon_full_round_chain_output_limb_27_col281,
        poseidon_full_round_chain_output_limb_28_col282,
        poseidon_full_round_chain_output_limb_29_col283,
        unpacked_limb_0_col322,
        unpacked_limb_1_col323,
        unpacked_limb_3_col324,
        unpacked_limb_4_col325,
        unpacked_limb_6_col326,
        unpacked_limb_7_col327,
        unpacked_limb_9_col328,
        unpacked_limb_10_col329,
        unpacked_limb_12_col330,
        unpacked_limb_13_col331,
        unpacked_limb_15_col332,
        unpacked_limb_16_col333,
        unpacked_limb_18_col334,
        unpacked_limb_19_col335,
        unpacked_limb_21_col336,
        unpacked_limb_22_col337,
        unpacked_limb_24_col338,
        unpacked_limb_25_col339,
        felt_252_unpack_output_2,
        &cuda_evaluator1
    );

    mem_verify_evaluate<EvaluatorT>(
        add(instance_addr, M31_5),
        unpacked_limb_0_col322,
        unpacked_limb_1_col323,
        felt_252_unpack_output_2[0],  // limb_2
        unpacked_limb_3_col324,
        unpacked_limb_4_col325,
        felt_252_unpack_output_2[1],  // limb_5
        unpacked_limb_6_col326,
        unpacked_limb_7_col327,
        felt_252_unpack_output_2[2],  // limb_8
        unpacked_limb_9_col328,
        unpacked_limb_10_col329,
        felt_252_unpack_output_2[3],  // limb_11
        unpacked_limb_12_col330,
        unpacked_limb_13_col331,
        felt_252_unpack_output_2[4],  // limb_14
        unpacked_limb_15_col332,
        unpacked_limb_16_col333,
        felt_252_unpack_output_2[5],  // limb_17
        unpacked_limb_18_col334,
        unpacked_limb_19_col335,
        felt_252_unpack_output_2[6],  // limb_20
        unpacked_limb_21_col336,
        unpacked_limb_22_col337,
        felt_252_unpack_output_2[7],  // limb_23
        unpacked_limb_24_col338,
        unpacked_limb_25_col339,
        felt_252_unpack_output_2[8],  // limb_26
        poseidon_full_round_chain_output_limb_29_col283,  // limb_27
        output_state_2_id_col340,
        poseidon_eval->memory_address_to_id_lookup_elements,
        poseidon_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== Complete constraint evaluation =====================
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// Poseidon Main Function
// ============================================================================

extern "C"
void evaluate_poseidon_builtin(
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
    bool use_assert_evaluator
) {
    (void)number_of_columns;

    PoseidonBuiltin_Eval *poseidon_eval =
        (PoseidonBuiltin_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // Clone trace0 and trace1 separately to device
    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PoseidonBuiltin_Eval *device_poseidon_eval =
        cuda_malloc<PoseidonBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<PoseidonBuiltin_Eval>(
        poseidon_eval, device_poseidon_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_poseidon_builtin");

    int block_dim = eval_domain_size < POSEIDON_BUILTIN_THREAD_COUNT_MAX
        ? eval_domain_size
        : POSEIDON_BUILTIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_poseidon_builtin_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_poseidon_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_poseidon_builtin_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_poseidon_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = logup_counts ? batching[logup_counts - 1] : 0;

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim>>>(
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
        generic_constraint_post_kernel<CudaEvaluator><<<
            num_blocks, block_dim>>>(
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

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        should_accumulate
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_poseidon_builtin");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_poseidon_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
