// ============================================================================
// Pedersen Builtin CUDA Evaluator
// ============================================================================
//
// CUDA version of Pedersen builtin AIR constraint evaluator
// Translated from cairo-air/src/components/pedersen_builtin.rs
//
// ## Overview
// Pedersen builtin performs Pedersen hash operations through elliptic curve
// scalar multiplication. It reads two field elements (a, b) from memory,
// computes the Pedersen hash, and writes the result back to memory.
//
// ## Data Structure
// - 351 trace columns:
//   - Columns 0-28: First input value (a) with 27 limbs + ms_limb split
//   - Columns 29-58: pedersen_a_id and second input value (b)
//   - Columns 59-65: Range check flags for reduced verification
//   - Columns 66-207: First PartialEcMul output (142 limbs)
//   - Columns 208-349: Second through fourth PartialEcMul outputs
//   - Column 350: pedersen_result_id
//
// ## Constraint Logic
// 1. ReadSplit: Read two 252-bit values from memory (a and b)
// 2. VerifyReduced252: Verify both inputs are properly reduced
// 3. PartialEcMul: 4 EC multiplication relation lookups
// 4. MemVerify: Verify result written to memory
//
// ## Relation Lookups
// - MemoryAddressToId: 3 uses (2 from ReadSplit, 1 from MemVerify)
// - MemoryIdToBig: 3 uses (2 from ReadSplit, 1 from MemVerify)
// - PartialEcMul: 4 uses (EC multiplication chain)
// - RangeCheck_5_4: 2 uses (from ReadSplit)
// - RangeCheck_8: 4 uses (2 from each VerifyReduced252)
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
#include "evaluate_pedersen_builtin.cuh"
#include "evaluate_read_split.cuh"
#include "evaluate_verify_reduced_252.cuh"
#include "evaluate_mem_verify.cuh"
#include "evaluate_common.cuh"

#define PEDERSEN_BUILTIN_THREAD_COUNT_MAX 256

// ============================================================================
// Pedersen Pre-Kernel: Main constraint evaluation
// ============================================================================
// This kernel is responsible for:
// 1. Reading all 351 trace columns
// 2. Calling ReadSplit for both inputs (a, b)
// 3. Calling VerifyReduced252 for both inputs
// 4. Adding 4 PartialEcMul relation lookups
// 5. Calling MemVerify for output
// 6. Accumulating all constraints to numerators
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_pedersen_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PedersenBuiltin_Eval *pedersen_eval,
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

    // Evaluator for preprocessed trace (trace0)
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

    // Read preprocessed column (Seq)
    m31 seq = cuda_evaluator0.next_trace_mask();

    // Evaluator for base trace (trace1)
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

    // ===================== Read all 351 trace columns =====================
    // Columns 0-26: First input value limbs (27 limbs, 9 bits each)
    m31 value_limb_0_col0 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_1_col1 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_2_col2 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_3_col3 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_4_col4 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_5_col5 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_6_col6 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_7_col7 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_8_col8 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_9_col9 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_10_col10 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_11_col11 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_12_col12 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_13_col13 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_14_col14 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_15_col15 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_16_col16 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_17_col17 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_18_col18 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_19_col19 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_20_col20 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_21_col21 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_22_col22 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_23_col23 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_24_col24 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_25_col25 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_26_col26 = cuda_evaluator1.next_trace_mask();

    // Columns 27-29: MS limb split for first input + memory ID
    m31 ms_limb_low_col27 = cuda_evaluator1.next_trace_mask();
    m31 ms_limb_high_col28 = cuda_evaluator1.next_trace_mask();
    m31 pedersen_a_id_col29 = cuda_evaluator1.next_trace_mask();

    // Columns 30-56: Second input value limbs (27 limbs, 9 bits each)
    m31 value_limb_0_col30 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_1_col31 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_2_col32 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_3_col33 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_4_col34 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_5_col35 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_6_col36 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_7_col37 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_8_col38 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_9_col39 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_10_col40 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_11_col41 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_12_col42 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_13_col43 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_14_col44 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_15_col45 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_16_col46 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_17_col47 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_18_col48 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_19_col49 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_20_col50 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_21_col51 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_22_col52 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_23_col53 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_24_col54 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_25_col55 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_26_col56 = cuda_evaluator1.next_trace_mask();

    // Columns 57-59: MS limb split for second input + memory ID
    m31 ms_limb_low_col57 = cuda_evaluator1.next_trace_mask();
    m31 ms_limb_high_col58 = cuda_evaluator1.next_trace_mask();
    m31 pedersen_b_id_col59 = cuda_evaluator1.next_trace_mask();

    // Columns 60-65: VerifyReduced252 auxiliary columns
    m31 ms_limb_is_max_col60 = cuda_evaluator1.next_trace_mask();
    m31 ms_and_mid_limbs_are_max_col61 = cuda_evaluator1.next_trace_mask();
    m31 rc_input_col62 = cuda_evaluator1.next_trace_mask();
    m31 ms_limb_is_max_col63 = cuda_evaluator1.next_trace_mask();
    m31 ms_and_mid_limbs_are_max_col64 = cuda_evaluator1.next_trace_mask();
    m31 rc_input_col65 = cuda_evaluator1.next_trace_mask();

    // Columns 66-136: First PartialEcMul output (71 limbs)
    m31 partial_ec_mul_output_limb_0_col66 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_1_col67 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_2_col68 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_3_col69 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_4_col70 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_5_col71 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_6_col72 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_7_col73 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_8_col74 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_9_col75 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_10_col76 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_11_col77 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_12_col78 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_13_col79 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_14_col80 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_15_col81 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_16_col82 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_17_col83 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_18_col84 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_19_col85 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_20_col86 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_21_col87 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_22_col88 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_23_col89 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_24_col90 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_25_col91 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_26_col92 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_27_col93 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_28_col94 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_29_col95 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_30_col96 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_31_col97 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_32_col98 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_33_col99 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_34_col100 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_35_col101 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_36_col102 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_37_col103 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_38_col104 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_39_col105 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_40_col106 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_41_col107 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_42_col108 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_43_col109 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_44_col110 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_45_col111 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_46_col112 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_47_col113 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_48_col114 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_49_col115 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_50_col116 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_51_col117 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_52_col118 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_53_col119 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_54_col120 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_55_col121 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_56_col122 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_57_col123 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_58_col124 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_59_col125 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_60_col126 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_61_col127 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_62_col128 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_63_col129 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_64_col130 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_65_col131 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_66_col132 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_67_col133 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_68_col134 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_69_col135 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_70_col136 = cuda_evaluator1.next_trace_mask();

    // Columns 137-207: Second PartialEcMul output (71 limbs)
    m31 partial_ec_mul_output_limb_0_col137 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_1_col138 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_2_col139 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_3_col140 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_4_col141 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_5_col142 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_6_col143 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_7_col144 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_8_col145 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_9_col146 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_10_col147 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_11_col148 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_12_col149 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_13_col150 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_14_col151 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_15_col152 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_16_col153 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_17_col154 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_18_col155 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_19_col156 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_20_col157 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_21_col158 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_22_col159 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_23_col160 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_24_col161 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_25_col162 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_26_col163 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_27_col164 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_28_col165 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_29_col166 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_30_col167 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_31_col168 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_32_col169 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_33_col170 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_34_col171 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_35_col172 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_36_col173 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_37_col174 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_38_col175 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_39_col176 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_40_col177 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_41_col178 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_42_col179 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_43_col180 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_44_col181 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_45_col182 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_46_col183 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_47_col184 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_48_col185 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_49_col186 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_50_col187 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_51_col188 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_52_col189 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_53_col190 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_54_col191 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_55_col192 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_56_col193 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_57_col194 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_58_col195 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_59_col196 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_60_col197 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_61_col198 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_62_col199 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_63_col200 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_64_col201 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_65_col202 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_66_col203 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_67_col204 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_68_col205 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_69_col206 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_70_col207 = cuda_evaluator1.next_trace_mask();

    // Columns 208-278: Third PartialEcMul output (71 limbs)
    m31 partial_ec_mul_output_limb_0_col208 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_1_col209 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_2_col210 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_3_col211 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_4_col212 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_5_col213 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_6_col214 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_7_col215 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_8_col216 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_9_col217 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_10_col218 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_11_col219 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_12_col220 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_13_col221 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_14_col222 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_15_col223 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_16_col224 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_17_col225 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_18_col226 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_19_col227 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_20_col228 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_21_col229 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_22_col230 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_23_col231 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_24_col232 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_25_col233 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_26_col234 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_27_col235 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_28_col236 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_29_col237 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_30_col238 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_31_col239 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_32_col240 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_33_col241 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_34_col242 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_35_col243 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_36_col244 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_37_col245 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_38_col246 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_39_col247 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_40_col248 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_41_col249 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_42_col250 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_43_col251 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_44_col252 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_45_col253 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_46_col254 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_47_col255 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_48_col256 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_49_col257 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_50_col258 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_51_col259 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_52_col260 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_53_col261 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_54_col262 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_55_col263 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_56_col264 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_57_col265 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_58_col266 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_59_col267 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_60_col268 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_61_col269 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_62_col270 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_63_col271 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_64_col272 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_65_col273 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_66_col274 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_67_col275 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_68_col276 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_69_col277 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_70_col278 = cuda_evaluator1.next_trace_mask();

    // Columns 279-349: Fourth PartialEcMul output (71 limbs)
    m31 partial_ec_mul_output_limb_0_col279 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_1_col280 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_2_col281 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_3_col282 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_4_col283 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_5_col284 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_6_col285 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_7_col286 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_8_col287 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_9_col288 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_10_col289 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_11_col290 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_12_col291 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_13_col292 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_14_col293 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_15_col294 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_16_col295 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_17_col296 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_18_col297 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_19_col298 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_20_col299 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_21_col300 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_22_col301 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_23_col302 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_24_col303 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_25_col304 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_26_col305 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_27_col306 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_28_col307 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_29_col308 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_30_col309 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_31_col310 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_32_col311 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_33_col312 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_34_col313 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_35_col314 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_36_col315 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_37_col316 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_38_col317 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_39_col318 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_40_col319 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_41_col320 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_42_col321 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_43_col322 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_44_col323 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_45_col324 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_46_col325 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_47_col326 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_48_col327 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_49_col328 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_50_col329 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_51_col330 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_52_col331 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_53_col332 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_54_col333 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_55_col334 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_56_col335 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_57_col336 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_58_col337 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_59_col338 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_60_col339 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_61_col340 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_62_col341 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_63_col342 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_64_col343 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_65_col344 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_66_col345 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_67_col346 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_68_col347 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_69_col348 = cuda_evaluator1.next_trace_mask();
    m31 partial_ec_mul_output_limb_70_col349 = cuda_evaluator1.next_trace_mask();

    // Column 350: Result memory ID
    m31 pedersen_result_id_col350 = cuda_evaluator1.next_trace_mask();

    // ===================== Constants =====================
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_3 = m31(3);
    const m31 M31_4 = m31(4);
    const m31 M31_14 = m31(14);
    const m31 M31_15 = m31(15);
    const m31 M31_24 = m31(24);
    const m31 M31_27 = m31(27);
    const m31 M31_31 = m31(31);
    const m31 M31_50 = m31(50);
    const m31 M31_83 = m31(83);
    const m31 M31_92 = m31(92);
    const m31 M31_96 = m31(96);
    const m31 M31_102 = m31(102);
    const m31 M31_118 = m31(118);
    const m31 M31_125 = m31(125);
    const m31 M31_130 = m31(130);
    const m31 M31_145 = m31(145);
    const m31 M31_191 = m31(191);
    const m31 M31_202 = m31(202);
    const m31 M31_212 = m31(212);
    const m31 M31_213 = m31(213);
    const m31 M31_221 = m31(221);
    const m31 M31_222 = m31(222);
    const m31 M31_226 = m31(226);
    const m31 M31_227 = m31(227);
    const m31 M31_228 = m31(228);
    const m31 M31_251 = m31(251);
    const m31 M31_252 = m31(252);
    const m31 M31_253 = m31(253);
    const m31 M31_259 = m31(259);
    const m31 M31_264 = m31(264);
    const m31 M31_269 = m31(269);
    const m31 M31_276 = m31(276);
    const m31 M31_281 = m31(281);
    const m31 M31_301 = m31(301);
    const m31 M31_308 = m31(308);
    const m31 M31_319 = m31(319);
    const m31 M31_321 = m31(321);
    const m31 M31_330 = m31(330);
    const m31 M31_334 = m31(334);
    const m31 M31_354 = m31(354);
    const m31 M31_377 = m31(377);
    const m31 M31_383 = m31(383);
    const m31 M31_385 = m31(385);
    const m31 M31_413 = m31(413);
    const m31 M31_419 = m31(419);
    const m31 M31_422 = m31(422);
    const m31 M31_435 = m31(435);
    const m31 M31_458 = m31(458);
    const m31 M31_461 = m31(461);
    const m31 M31_464 = m31(464);
    const m31 M31_471 = m31(471);
    const m31 M31_472 = m31(472);
    const m31 M31_483 = m31(483);
    const m31 M31_508 = m31(508);
    const m31 M31_512 = m31(512);
    const m31 M31_3670016 = m31(3670016);
    const m31 M31_3670032 = m31(3670032);
    const m31 M31_7340048 = m31(7340048);

    // ===================== Compute instance address =====================
    // instance_addr = seq * 3 + pedersen_builtin_segment_start
    m31 instance_addr = add(
        mul(seq, M31_3),
        m31(pedersen_eval->claim.pedersen_builtin_segment_start)
    );

    // ===================== ReadSplit: First input (a) =====================
    // Reads first 252-bit value from memory and verifies the MS limb split
    m31 read_split_output_limb_27_a = read_split_evaluate<EvaluatorT>(
        instance_addr,
        value_limb_0_col0, value_limb_1_col1, value_limb_2_col2, value_limb_3_col3,
        value_limb_4_col4, value_limb_5_col5, value_limb_6_col6, value_limb_7_col7,
        value_limb_8_col8, value_limb_9_col9, value_limb_10_col10, value_limb_11_col11,
        value_limb_12_col12, value_limb_13_col13, value_limb_14_col14, value_limb_15_col15,
        value_limb_16_col16, value_limb_17_col17, value_limb_18_col18, value_limb_19_col19,
        value_limb_20_col20, value_limb_21_col21, value_limb_22_col22, value_limb_23_col23,
        value_limb_24_col24, value_limb_25_col25, value_limb_26_col26,
        ms_limb_low_col27, ms_limb_high_col28, pedersen_a_id_col29,
        pedersen_eval->range_check_5_4_lookup_elements,
        pedersen_eval->memory_address_to_id_lookup_elements,
        pedersen_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== ReadSplit: Second input (b) =====================
    // Reads second 252-bit value from memory (address = instance_addr + 1)
    m31 read_split_output_limb_27_b = read_split_evaluate<EvaluatorT>(
        add(instance_addr, M31_1),
        value_limb_0_col30, value_limb_1_col31, value_limb_2_col32, value_limb_3_col33,
        value_limb_4_col34, value_limb_5_col35, value_limb_6_col36, value_limb_7_col37,
        value_limb_8_col38, value_limb_9_col39, value_limb_10_col40, value_limb_11_col41,
        value_limb_12_col42, value_limb_13_col43, value_limb_14_col44, value_limb_15_col45,
        value_limb_16_col46, value_limb_17_col47, value_limb_18_col48, value_limb_19_col49,
        value_limb_20_col50, value_limb_21_col51, value_limb_22_col52, value_limb_23_col53,
        value_limb_24_col54, value_limb_25_col55, value_limb_26_col56,
        ms_limb_low_col57, ms_limb_high_col58, pedersen_b_id_col59,
        pedersen_eval->range_check_5_4_lookup_elements,
        pedersen_eval->memory_address_to_id_lookup_elements,
        pedersen_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== VerifyReduced252: First input (a) =====================
    // Verifies that the first input is properly reduced (< field prime)
    verify_reduced_252_evaluate<EvaluatorT>(
        value_limb_0_col0, value_limb_1_col1, value_limb_2_col2, value_limb_3_col3,
        value_limb_4_col4, value_limb_5_col5, value_limb_6_col6, value_limb_7_col7,
        value_limb_8_col8, value_limb_9_col9, value_limb_10_col10, value_limb_11_col11,
        value_limb_12_col12, value_limb_13_col13, value_limb_14_col14, value_limb_15_col15,
        value_limb_16_col16, value_limb_17_col17, value_limb_18_col18, value_limb_19_col19,
        value_limb_20_col20, value_limb_21_col21, value_limb_22_col22, value_limb_23_col23,
        value_limb_24_col24, value_limb_25_col25, value_limb_26_col26,
        read_split_output_limb_27_a,
        ms_limb_is_max_col60, ms_and_mid_limbs_are_max_col61, rc_input_col62,
        pedersen_eval->range_check_8_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== VerifyReduced252: Second input (b) =====================
    // Verifies that the second input is properly reduced
    verify_reduced_252_evaluate<EvaluatorT>(
        value_limb_0_col30, value_limb_1_col31, value_limb_2_col32, value_limb_3_col33,
        value_limb_4_col34, value_limb_5_col35, value_limb_6_col36, value_limb_7_col37,
        value_limb_8_col38, value_limb_9_col39, value_limb_10_col40, value_limb_11_col41,
        value_limb_12_col42, value_limb_13_col43, value_limb_14_col44, value_limb_15_col45,
        value_limb_16_col46, value_limb_17_col47, value_limb_18_col48, value_limb_19_col49,
        value_limb_20_col50, value_limb_21_col51, value_limb_22_col52, value_limb_23_col53,
        value_limb_24_col54, value_limb_25_col55, value_limb_26_col56,
        read_split_output_limb_27_b,
        ms_limb_is_max_col63, ms_and_mid_limbs_are_max_col64, rc_input_col65,
        pedersen_eval->range_check_8_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== PartialEcMul: Calculate 4 chain IDs =====================
    // We have 4 EC multiplication chains, each with a unique chain_id
    m31 partial_ec_mul_chain_id_0 = mul(seq, M31_4);                          // base chain
    m31 partial_ec_mul_chain_id_1 = add(partial_ec_mul_chain_id_0, M31_1);    // base + 1
    m31 partial_ec_mul_chain_id_2 = add(partial_ec_mul_chain_id_0, M31_2);    // base + 2
    m31 partial_ec_mul_chain_id_3 = add(partial_ec_mul_chain_id_0, M31_3);    // base + 3

    // ===================== Pair 1, Lookup 1: chain_id_0, step=0, mult=-1 =====================
    // Input to first EC multiplication (scalar from first value + EC base point constants)
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_0;
        rel_values[1] = M31_0;  // step = 0
        rel_values[2] = M31_0;  // padding

        // Scalar limbs (combined pairs for 18-bit values)
        rel_values[3] = add(value_limb_0_col0, mul(value_limb_1_col1, M31_512));
        rel_values[4] = add(value_limb_2_col2, mul(value_limb_3_col3, M31_512));
        rel_values[5] = add(value_limb_4_col4, mul(value_limb_5_col5, M31_512));
        rel_values[6] = add(value_limb_6_col6, mul(value_limb_7_col7, M31_512));
        rel_values[7] = add(value_limb_8_col8, mul(value_limb_9_col9, M31_512));
        rel_values[8] = add(value_limb_10_col10, mul(value_limb_11_col11, M31_512));
        rel_values[9] = add(value_limb_12_col12, mul(value_limb_13_col13, M31_512));
        rel_values[10] = add(value_limb_14_col14, mul(value_limb_15_col15, M31_512));
        rel_values[11] = add(value_limb_16_col16, mul(value_limb_17_col17, M31_512));
        rel_values[12] = add(value_limb_18_col18, mul(value_limb_19_col19, M31_512));
        rel_values[13] = add(value_limb_20_col20, mul(value_limb_21_col21, M31_512));
        rel_values[14] = add(value_limb_22_col22, mul(value_limb_23_col23, M31_512));
        rel_values[15] = add(value_limb_24_col24, mul(value_limb_25_col25, M31_512));
        rel_values[16] = add(value_limb_26_col26, mul(ms_limb_low_col27, M31_512));

        // EC point base constant values (from pedersen_builtin.rs line 678-733)
        rel_values[17] = M31_435;
        rel_values[18] = M31_50;
        rel_values[19] = M31_508;
        rel_values[20] = M31_83;
        rel_values[21] = M31_221;
        rel_values[22] = M31_281;
        rel_values[23] = M31_377;
        rel_values[24] = M31_383;
        rel_values[25] = M31_212;
        rel_values[26] = M31_264;
        rel_values[27] = M31_301;
        rel_values[28] = M31_458;
        rel_values[29] = M31_130;
        rel_values[30] = M31_102;
        rel_values[31] = M31_385;
        rel_values[32] = M31_269;
        rel_values[33] = M31_145;
        rel_values[34] = M31_276;
        rel_values[35] = M31_483;
        rel_values[36] = M31_226;
        rel_values[37] = M31_422;
        rel_values[38] = M31_253;
        rel_values[39] = M31_308;
        rel_values[40] = M31_125;
        rel_values[41] = M31_472;
        rel_values[42] = M31_301;
        rel_values[43] = M31_227;
        rel_values[44] = M31_27;
        rel_values[45] = M31_92;
        rel_values[46] = M31_321;
        rel_values[47] = M31_252;
        rel_values[48] = M31_259;
        rel_values[49] = M31_252;
        rel_values[50] = M31_413;
        rel_values[51] = M31_228;
        rel_values[52] = M31_31;
        rel_values[53] = M31_24;
        rel_values[54] = M31_118;
        rel_values[55] = M31_301;
        rel_values[56] = M31_202;
        rel_values[57] = M31_15;
        rel_values[58] = M31_464;
        rel_values[59] = M31_334;
        rel_values[60] = M31_212;
        rel_values[61] = M31_471;
        rel_values[62] = M31_461;
        rel_values[63] = M31_419;
        rel_values[64] = M31_354;
        rel_values[65] = M31_96;
        rel_values[66] = M31_213;
        rel_values[67] = M31_319;
        rel_values[68] = M31_191;
        rel_values[69] = M31_251;
        rel_values[70] = M31_330;
        rel_values[71] = M31_15;
        rel_values[72] = M31_222;

        // Negative multiplicity for PROVIDE side (use neg(1) to avoid sign conversion warning)
        RelationEntry<73> ec_mul_entry_1(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{neg(m31(1)), 0}, {0, 0}},  // negative multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_1);
    }

    // ===================== Pair 1, Lookup 2: chain_id_0, step=14, mult=+1 =====================
    // Output from first EC multiplication at step 14
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_0;
        rel_values[1] = M31_14;  // step = 14

        // Explicitly list all 71 output limbs from columns 66-136 (safer than pointer arithmetic)
        rel_values[2] = partial_ec_mul_output_limb_0_col66;
        rel_values[3] = partial_ec_mul_output_limb_1_col67;
        rel_values[4] = partial_ec_mul_output_limb_2_col68;
        rel_values[5] = partial_ec_mul_output_limb_3_col69;
        rel_values[6] = partial_ec_mul_output_limb_4_col70;
        rel_values[7] = partial_ec_mul_output_limb_5_col71;
        rel_values[8] = partial_ec_mul_output_limb_6_col72;
        rel_values[9] = partial_ec_mul_output_limb_7_col73;
        rel_values[10] = partial_ec_mul_output_limb_8_col74;
        rel_values[11] = partial_ec_mul_output_limb_9_col75;
        rel_values[12] = partial_ec_mul_output_limb_10_col76;
        rel_values[13] = partial_ec_mul_output_limb_11_col77;
        rel_values[14] = partial_ec_mul_output_limb_12_col78;
        rel_values[15] = partial_ec_mul_output_limb_13_col79;
        rel_values[16] = partial_ec_mul_output_limb_14_col80;
        rel_values[17] = partial_ec_mul_output_limb_15_col81;
        rel_values[18] = partial_ec_mul_output_limb_16_col82;
        rel_values[19] = partial_ec_mul_output_limb_17_col83;
        rel_values[20] = partial_ec_mul_output_limb_18_col84;
        rel_values[21] = partial_ec_mul_output_limb_19_col85;
        rel_values[22] = partial_ec_mul_output_limb_20_col86;
        rel_values[23] = partial_ec_mul_output_limb_21_col87;
        rel_values[24] = partial_ec_mul_output_limb_22_col88;
        rel_values[25] = partial_ec_mul_output_limb_23_col89;
        rel_values[26] = partial_ec_mul_output_limb_24_col90;
        rel_values[27] = partial_ec_mul_output_limb_25_col91;
        rel_values[28] = partial_ec_mul_output_limb_26_col92;
        rel_values[29] = partial_ec_mul_output_limb_27_col93;
        rel_values[30] = partial_ec_mul_output_limb_28_col94;
        rel_values[31] = partial_ec_mul_output_limb_29_col95;
        rel_values[32] = partial_ec_mul_output_limb_30_col96;
        rel_values[33] = partial_ec_mul_output_limb_31_col97;
        rel_values[34] = partial_ec_mul_output_limb_32_col98;
        rel_values[35] = partial_ec_mul_output_limb_33_col99;
        rel_values[36] = partial_ec_mul_output_limb_34_col100;
        rel_values[37] = partial_ec_mul_output_limb_35_col101;
        rel_values[38] = partial_ec_mul_output_limb_36_col102;
        rel_values[39] = partial_ec_mul_output_limb_37_col103;
        rel_values[40] = partial_ec_mul_output_limb_38_col104;
        rel_values[41] = partial_ec_mul_output_limb_39_col105;
        rel_values[42] = partial_ec_mul_output_limb_40_col106;
        rel_values[43] = partial_ec_mul_output_limb_41_col107;
        rel_values[44] = partial_ec_mul_output_limb_42_col108;
        rel_values[45] = partial_ec_mul_output_limb_43_col109;
        rel_values[46] = partial_ec_mul_output_limb_44_col110;
        rel_values[47] = partial_ec_mul_output_limb_45_col111;
        rel_values[48] = partial_ec_mul_output_limb_46_col112;
        rel_values[49] = partial_ec_mul_output_limb_47_col113;
        rel_values[50] = partial_ec_mul_output_limb_48_col114;
        rel_values[51] = partial_ec_mul_output_limb_49_col115;
        rel_values[52] = partial_ec_mul_output_limb_50_col116;
        rel_values[53] = partial_ec_mul_output_limb_51_col117;
        rel_values[54] = partial_ec_mul_output_limb_52_col118;
        rel_values[55] = partial_ec_mul_output_limb_53_col119;
        rel_values[56] = partial_ec_mul_output_limb_54_col120;
        rel_values[57] = partial_ec_mul_output_limb_55_col121;
        rel_values[58] = partial_ec_mul_output_limb_56_col122;
        rel_values[59] = partial_ec_mul_output_limb_57_col123;
        rel_values[60] = partial_ec_mul_output_limb_58_col124;
        rel_values[61] = partial_ec_mul_output_limb_59_col125;
        rel_values[62] = partial_ec_mul_output_limb_60_col126;
        rel_values[63] = partial_ec_mul_output_limb_61_col127;
        rel_values[64] = partial_ec_mul_output_limb_62_col128;
        rel_values[65] = partial_ec_mul_output_limb_63_col129;
        rel_values[66] = partial_ec_mul_output_limb_64_col130;
        rel_values[67] = partial_ec_mul_output_limb_65_col131;
        rel_values[68] = partial_ec_mul_output_limb_66_col132;
        rel_values[69] = partial_ec_mul_output_limb_67_col133;
        rel_values[70] = partial_ec_mul_output_limb_68_col134;
        rel_values[71] = partial_ec_mul_output_limb_69_col135;
        rel_values[72] = partial_ec_mul_output_limb_70_col136;

        RelationEntry<73> ec_mul_entry_2(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{1, 0}, {0, 0}},  // positive multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_2);
    }

    // ===================== Pair 2, Lookup 3: chain_id_1, step=0, mult=-1 =====================
    // Input to second EC multiplication (ms_limb_high + output from previous step)
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_1;
        rel_values[1] = M31_0;  // step = 0
        rel_values[2] = M31_3670016;
        rel_values[3] = ms_limb_high_col28;

        // Zeros for limbs 4-16 (13 zeros, not 12!)
        for (int i = 4; i <= 16; i++) {
            rel_values[i] = M31_0;
        }

        // Limbs from col81-136 (partial_ec_mul_output_limb_15_col81 onwards)
        // Explicit assignments to avoid pointer arithmetic issues
        rel_values[17] = partial_ec_mul_output_limb_15_col81;
        rel_values[18] = partial_ec_mul_output_limb_16_col82;
        rel_values[19] = partial_ec_mul_output_limb_17_col83;
        rel_values[20] = partial_ec_mul_output_limb_18_col84;
        rel_values[21] = partial_ec_mul_output_limb_19_col85;
        rel_values[22] = partial_ec_mul_output_limb_20_col86;
        rel_values[23] = partial_ec_mul_output_limb_21_col87;
        rel_values[24] = partial_ec_mul_output_limb_22_col88;
        rel_values[25] = partial_ec_mul_output_limb_23_col89;
        rel_values[26] = partial_ec_mul_output_limb_24_col90;
        rel_values[27] = partial_ec_mul_output_limb_25_col91;
        rel_values[28] = partial_ec_mul_output_limb_26_col92;
        rel_values[29] = partial_ec_mul_output_limb_27_col93;
        rel_values[30] = partial_ec_mul_output_limb_28_col94;
        rel_values[31] = partial_ec_mul_output_limb_29_col95;
        rel_values[32] = partial_ec_mul_output_limb_30_col96;
        rel_values[33] = partial_ec_mul_output_limb_31_col97;
        rel_values[34] = partial_ec_mul_output_limb_32_col98;
        rel_values[35] = partial_ec_mul_output_limb_33_col99;
        rel_values[36] = partial_ec_mul_output_limb_34_col100;
        rel_values[37] = partial_ec_mul_output_limb_35_col101;
        rel_values[38] = partial_ec_mul_output_limb_36_col102;
        rel_values[39] = partial_ec_mul_output_limb_37_col103;
        rel_values[40] = partial_ec_mul_output_limb_38_col104;
        rel_values[41] = partial_ec_mul_output_limb_39_col105;
        rel_values[42] = partial_ec_mul_output_limb_40_col106;
        rel_values[43] = partial_ec_mul_output_limb_41_col107;
        rel_values[44] = partial_ec_mul_output_limb_42_col108;
        rel_values[45] = partial_ec_mul_output_limb_43_col109;
        rel_values[46] = partial_ec_mul_output_limb_44_col110;
        rel_values[47] = partial_ec_mul_output_limb_45_col111;
        rel_values[48] = partial_ec_mul_output_limb_46_col112;
        rel_values[49] = partial_ec_mul_output_limb_47_col113;
        rel_values[50] = partial_ec_mul_output_limb_48_col114;
        rel_values[51] = partial_ec_mul_output_limb_49_col115;
        rel_values[52] = partial_ec_mul_output_limb_50_col116;
        rel_values[53] = partial_ec_mul_output_limb_51_col117;
        rel_values[54] = partial_ec_mul_output_limb_52_col118;
        rel_values[55] = partial_ec_mul_output_limb_53_col119;
        rel_values[56] = partial_ec_mul_output_limb_54_col120;
        rel_values[57] = partial_ec_mul_output_limb_55_col121;
        rel_values[58] = partial_ec_mul_output_limb_56_col122;
        rel_values[59] = partial_ec_mul_output_limb_57_col123;
        rel_values[60] = partial_ec_mul_output_limb_58_col124;
        rel_values[61] = partial_ec_mul_output_limb_59_col125;
        rel_values[62] = partial_ec_mul_output_limb_60_col126;
        rel_values[63] = partial_ec_mul_output_limb_61_col127;
        rel_values[64] = partial_ec_mul_output_limb_62_col128;
        rel_values[65] = partial_ec_mul_output_limb_63_col129;
        rel_values[66] = partial_ec_mul_output_limb_64_col130;
        rel_values[67] = partial_ec_mul_output_limb_65_col131;
        rel_values[68] = partial_ec_mul_output_limb_66_col132;
        rel_values[69] = partial_ec_mul_output_limb_67_col133;
        rel_values[70] = partial_ec_mul_output_limb_68_col134;
        rel_values[71] = partial_ec_mul_output_limb_69_col135;
        rel_values[72] = partial_ec_mul_output_limb_70_col136;

        RelationEntry<73> ec_mul_entry_3(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{neg(m31(1)), 0}, {0, 0}},  // negative multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_3);
    }

    // ===================== Pair 2, Lookup 4: chain_id_1, step=1, mult=+1 =====================
    // Output from second EC multiplication at step 1
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_1;
        rel_values[1] = M31_1;  // step = 1

        // Explicitly list all 71 output limbs from columns 137-207
        rel_values[2] = partial_ec_mul_output_limb_0_col137;
        rel_values[3] = partial_ec_mul_output_limb_1_col138;
        rel_values[4] = partial_ec_mul_output_limb_2_col139;
        rel_values[5] = partial_ec_mul_output_limb_3_col140;
        rel_values[6] = partial_ec_mul_output_limb_4_col141;
        rel_values[7] = partial_ec_mul_output_limb_5_col142;
        rel_values[8] = partial_ec_mul_output_limb_6_col143;
        rel_values[9] = partial_ec_mul_output_limb_7_col144;
        rel_values[10] = partial_ec_mul_output_limb_8_col145;
        rel_values[11] = partial_ec_mul_output_limb_9_col146;
        rel_values[12] = partial_ec_mul_output_limb_10_col147;
        rel_values[13] = partial_ec_mul_output_limb_11_col148;
        rel_values[14] = partial_ec_mul_output_limb_12_col149;
        rel_values[15] = partial_ec_mul_output_limb_13_col150;
        rel_values[16] = partial_ec_mul_output_limb_14_col151;
        rel_values[17] = partial_ec_mul_output_limb_15_col152;
        rel_values[18] = partial_ec_mul_output_limb_16_col153;
        rel_values[19] = partial_ec_mul_output_limb_17_col154;
        rel_values[20] = partial_ec_mul_output_limb_18_col155;
        rel_values[21] = partial_ec_mul_output_limb_19_col156;
        rel_values[22] = partial_ec_mul_output_limb_20_col157;
        rel_values[23] = partial_ec_mul_output_limb_21_col158;
        rel_values[24] = partial_ec_mul_output_limb_22_col159;
        rel_values[25] = partial_ec_mul_output_limb_23_col160;
        rel_values[26] = partial_ec_mul_output_limb_24_col161;
        rel_values[27] = partial_ec_mul_output_limb_25_col162;
        rel_values[28] = partial_ec_mul_output_limb_26_col163;
        rel_values[29] = partial_ec_mul_output_limb_27_col164;
        rel_values[30] = partial_ec_mul_output_limb_28_col165;
        rel_values[31] = partial_ec_mul_output_limb_29_col166;
        rel_values[32] = partial_ec_mul_output_limb_30_col167;
        rel_values[33] = partial_ec_mul_output_limb_31_col168;
        rel_values[34] = partial_ec_mul_output_limb_32_col169;
        rel_values[35] = partial_ec_mul_output_limb_33_col170;
        rel_values[36] = partial_ec_mul_output_limb_34_col171;
        rel_values[37] = partial_ec_mul_output_limb_35_col172;
        rel_values[38] = partial_ec_mul_output_limb_36_col173;
        rel_values[39] = partial_ec_mul_output_limb_37_col174;
        rel_values[40] = partial_ec_mul_output_limb_38_col175;
        rel_values[41] = partial_ec_mul_output_limb_39_col176;
        rel_values[42] = partial_ec_mul_output_limb_40_col177;
        rel_values[43] = partial_ec_mul_output_limb_41_col178;
        rel_values[44] = partial_ec_mul_output_limb_42_col179;
        rel_values[45] = partial_ec_mul_output_limb_43_col180;
        rel_values[46] = partial_ec_mul_output_limb_44_col181;
        rel_values[47] = partial_ec_mul_output_limb_45_col182;
        rel_values[48] = partial_ec_mul_output_limb_46_col183;
        rel_values[49] = partial_ec_mul_output_limb_47_col184;
        rel_values[50] = partial_ec_mul_output_limb_48_col185;
        rel_values[51] = partial_ec_mul_output_limb_49_col186;
        rel_values[52] = partial_ec_mul_output_limb_50_col187;
        rel_values[53] = partial_ec_mul_output_limb_51_col188;
        rel_values[54] = partial_ec_mul_output_limb_52_col189;
        rel_values[55] = partial_ec_mul_output_limb_53_col190;
        rel_values[56] = partial_ec_mul_output_limb_54_col191;
        rel_values[57] = partial_ec_mul_output_limb_55_col192;
        rel_values[58] = partial_ec_mul_output_limb_56_col193;
        rel_values[59] = partial_ec_mul_output_limb_57_col194;
        rel_values[60] = partial_ec_mul_output_limb_58_col195;
        rel_values[61] = partial_ec_mul_output_limb_59_col196;
        rel_values[62] = partial_ec_mul_output_limb_60_col197;
        rel_values[63] = partial_ec_mul_output_limb_61_col198;
        rel_values[64] = partial_ec_mul_output_limb_62_col199;
        rel_values[65] = partial_ec_mul_output_limb_63_col200;
        rel_values[66] = partial_ec_mul_output_limb_64_col201;
        rel_values[67] = partial_ec_mul_output_limb_65_col202;
        rel_values[68] = partial_ec_mul_output_limb_66_col203;
        rel_values[69] = partial_ec_mul_output_limb_67_col204;
        rel_values[70] = partial_ec_mul_output_limb_68_col205;
        rel_values[71] = partial_ec_mul_output_limb_69_col206;
        rel_values[72] = partial_ec_mul_output_limb_70_col207;

        RelationEntry<73> ec_mul_entry_4(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{1, 0}, {0, 0}},  // positive multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_4);
    }

    // ===================== Pair 3, Lookup 5: chain_id_2, step=0, mult=-1 =====================
    // Input to third EC multiplication (scalar from second input + output from previous step)
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_2;
        rel_values[1] = M31_0;  // step = 0
        rel_values[2] = M31_3670032;

        // Scalar limbs from second input (col30-57 combined in pairs)
        rel_values[3] = add(value_limb_0_col30, mul(value_limb_1_col31, M31_512));
        rel_values[4] = add(value_limb_2_col32, mul(value_limb_3_col33, M31_512));
        rel_values[5] = add(value_limb_4_col34, mul(value_limb_5_col35, M31_512));
        rel_values[6] = add(value_limb_6_col36, mul(value_limb_7_col37, M31_512));
        rel_values[7] = add(value_limb_8_col38, mul(value_limb_9_col39, M31_512));
        rel_values[8] = add(value_limb_10_col40, mul(value_limb_11_col41, M31_512));
        rel_values[9] = add(value_limb_12_col42, mul(value_limb_13_col43, M31_512));
        rel_values[10] = add(value_limb_14_col44, mul(value_limb_15_col45, M31_512));
        rel_values[11] = add(value_limb_16_col46, mul(value_limb_17_col47, M31_512));
        rel_values[12] = add(value_limb_18_col48, mul(value_limb_19_col49, M31_512));
        rel_values[13] = add(value_limb_20_col50, mul(value_limb_21_col51, M31_512));
        rel_values[14] = add(value_limb_22_col52, mul(value_limb_23_col53, M31_512));
        rel_values[15] = add(value_limb_24_col54, mul(value_limb_25_col55, M31_512));
        rel_values[16] = add(value_limb_26_col56, mul(ms_limb_low_col57, M31_512));

        // Limbs from col152-207 (partial_ec_mul_output_limb_15_col152 onwards)
        // Explicit assignments to avoid pointer arithmetic issues
        rel_values[17] = partial_ec_mul_output_limb_15_col152;
        rel_values[18] = partial_ec_mul_output_limb_16_col153;
        rel_values[19] = partial_ec_mul_output_limb_17_col154;
        rel_values[20] = partial_ec_mul_output_limb_18_col155;
        rel_values[21] = partial_ec_mul_output_limb_19_col156;
        rel_values[22] = partial_ec_mul_output_limb_20_col157;
        rel_values[23] = partial_ec_mul_output_limb_21_col158;
        rel_values[24] = partial_ec_mul_output_limb_22_col159;
        rel_values[25] = partial_ec_mul_output_limb_23_col160;
        rel_values[26] = partial_ec_mul_output_limb_24_col161;
        rel_values[27] = partial_ec_mul_output_limb_25_col162;
        rel_values[28] = partial_ec_mul_output_limb_26_col163;
        rel_values[29] = partial_ec_mul_output_limb_27_col164;
        rel_values[30] = partial_ec_mul_output_limb_28_col165;
        rel_values[31] = partial_ec_mul_output_limb_29_col166;
        rel_values[32] = partial_ec_mul_output_limb_30_col167;
        rel_values[33] = partial_ec_mul_output_limb_31_col168;
        rel_values[34] = partial_ec_mul_output_limb_32_col169;
        rel_values[35] = partial_ec_mul_output_limb_33_col170;
        rel_values[36] = partial_ec_mul_output_limb_34_col171;
        rel_values[37] = partial_ec_mul_output_limb_35_col172;
        rel_values[38] = partial_ec_mul_output_limb_36_col173;
        rel_values[39] = partial_ec_mul_output_limb_37_col174;
        rel_values[40] = partial_ec_mul_output_limb_38_col175;
        rel_values[41] = partial_ec_mul_output_limb_39_col176;
        rel_values[42] = partial_ec_mul_output_limb_40_col177;
        rel_values[43] = partial_ec_mul_output_limb_41_col178;
        rel_values[44] = partial_ec_mul_output_limb_42_col179;
        rel_values[45] = partial_ec_mul_output_limb_43_col180;
        rel_values[46] = partial_ec_mul_output_limb_44_col181;
        rel_values[47] = partial_ec_mul_output_limb_45_col182;
        rel_values[48] = partial_ec_mul_output_limb_46_col183;
        rel_values[49] = partial_ec_mul_output_limb_47_col184;
        rel_values[50] = partial_ec_mul_output_limb_48_col185;
        rel_values[51] = partial_ec_mul_output_limb_49_col186;
        rel_values[52] = partial_ec_mul_output_limb_50_col187;
        rel_values[53] = partial_ec_mul_output_limb_51_col188;
        rel_values[54] = partial_ec_mul_output_limb_52_col189;
        rel_values[55] = partial_ec_mul_output_limb_53_col190;
        rel_values[56] = partial_ec_mul_output_limb_54_col191;
        rel_values[57] = partial_ec_mul_output_limb_55_col192;
        rel_values[58] = partial_ec_mul_output_limb_56_col193;
        rel_values[59] = partial_ec_mul_output_limb_57_col194;
        rel_values[60] = partial_ec_mul_output_limb_58_col195;
        rel_values[61] = partial_ec_mul_output_limb_59_col196;
        rel_values[62] = partial_ec_mul_output_limb_60_col197;
        rel_values[63] = partial_ec_mul_output_limb_61_col198;
        rel_values[64] = partial_ec_mul_output_limb_62_col199;
        rel_values[65] = partial_ec_mul_output_limb_63_col200;
        rel_values[66] = partial_ec_mul_output_limb_64_col201;
        rel_values[67] = partial_ec_mul_output_limb_65_col202;
        rel_values[68] = partial_ec_mul_output_limb_66_col203;
        rel_values[69] = partial_ec_mul_output_limb_67_col204;
        rel_values[70] = partial_ec_mul_output_limb_68_col205;
        rel_values[71] = partial_ec_mul_output_limb_69_col206;
        rel_values[72] = partial_ec_mul_output_limb_70_col207;

        RelationEntry<73> ec_mul_entry_5(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{neg(m31(1)), 0}, {0, 0}},  // negative multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_5);
    }

    // ===================== Pair 3, Lookup 6: chain_id_2, step=14, mult=+1 =====================
    // Output from third EC multiplication at step 14
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_2;
        rel_values[1] = M31_14;  // step = 14
        rel_values[2] = partial_ec_mul_output_limb_0_col208;

        // Output limbs from columns 208-278
        // Explicit assignments to avoid pointer arithmetic issues
        rel_values[3] = partial_ec_mul_output_limb_1_col209;
        rel_values[4] = partial_ec_mul_output_limb_2_col210;
        rel_values[5] = partial_ec_mul_output_limb_3_col211;
        rel_values[6] = partial_ec_mul_output_limb_4_col212;
        rel_values[7] = partial_ec_mul_output_limb_5_col213;
        rel_values[8] = partial_ec_mul_output_limb_6_col214;
        rel_values[9] = partial_ec_mul_output_limb_7_col215;
        rel_values[10] = partial_ec_mul_output_limb_8_col216;
        rel_values[11] = partial_ec_mul_output_limb_9_col217;
        rel_values[12] = partial_ec_mul_output_limb_10_col218;
        rel_values[13] = partial_ec_mul_output_limb_11_col219;
        rel_values[14] = partial_ec_mul_output_limb_12_col220;
        rel_values[15] = partial_ec_mul_output_limb_13_col221;
        rel_values[16] = partial_ec_mul_output_limb_14_col222;
        rel_values[17] = partial_ec_mul_output_limb_15_col223;
        rel_values[18] = partial_ec_mul_output_limb_16_col224;
        rel_values[19] = partial_ec_mul_output_limb_17_col225;
        rel_values[20] = partial_ec_mul_output_limb_18_col226;
        rel_values[21] = partial_ec_mul_output_limb_19_col227;
        rel_values[22] = partial_ec_mul_output_limb_20_col228;
        rel_values[23] = partial_ec_mul_output_limb_21_col229;
        rel_values[24] = partial_ec_mul_output_limb_22_col230;
        rel_values[25] = partial_ec_mul_output_limb_23_col231;
        rel_values[26] = partial_ec_mul_output_limb_24_col232;
        rel_values[27] = partial_ec_mul_output_limb_25_col233;
        rel_values[28] = partial_ec_mul_output_limb_26_col234;
        rel_values[29] = partial_ec_mul_output_limb_27_col235;
        rel_values[30] = partial_ec_mul_output_limb_28_col236;
        rel_values[31] = partial_ec_mul_output_limb_29_col237;
        rel_values[32] = partial_ec_mul_output_limb_30_col238;
        rel_values[33] = partial_ec_mul_output_limb_31_col239;
        rel_values[34] = partial_ec_mul_output_limb_32_col240;
        rel_values[35] = partial_ec_mul_output_limb_33_col241;
        rel_values[36] = partial_ec_mul_output_limb_34_col242;
        rel_values[37] = partial_ec_mul_output_limb_35_col243;
        rel_values[38] = partial_ec_mul_output_limb_36_col244;
        rel_values[39] = partial_ec_mul_output_limb_37_col245;
        rel_values[40] = partial_ec_mul_output_limb_38_col246;
        rel_values[41] = partial_ec_mul_output_limb_39_col247;
        rel_values[42] = partial_ec_mul_output_limb_40_col248;
        rel_values[43] = partial_ec_mul_output_limb_41_col249;
        rel_values[44] = partial_ec_mul_output_limb_42_col250;
        rel_values[45] = partial_ec_mul_output_limb_43_col251;
        rel_values[46] = partial_ec_mul_output_limb_44_col252;
        rel_values[47] = partial_ec_mul_output_limb_45_col253;
        rel_values[48] = partial_ec_mul_output_limb_46_col254;
        rel_values[49] = partial_ec_mul_output_limb_47_col255;
        rel_values[50] = partial_ec_mul_output_limb_48_col256;
        rel_values[51] = partial_ec_mul_output_limb_49_col257;
        rel_values[52] = partial_ec_mul_output_limb_50_col258;
        rel_values[53] = partial_ec_mul_output_limb_51_col259;
        rel_values[54] = partial_ec_mul_output_limb_52_col260;
        rel_values[55] = partial_ec_mul_output_limb_53_col261;
        rel_values[56] = partial_ec_mul_output_limb_54_col262;
        rel_values[57] = partial_ec_mul_output_limb_55_col263;
        rel_values[58] = partial_ec_mul_output_limb_56_col264;
        rel_values[59] = partial_ec_mul_output_limb_57_col265;
        rel_values[60] = partial_ec_mul_output_limb_58_col266;
        rel_values[61] = partial_ec_mul_output_limb_59_col267;
        rel_values[62] = partial_ec_mul_output_limb_60_col268;
        rel_values[63] = partial_ec_mul_output_limb_61_col269;
        rel_values[64] = partial_ec_mul_output_limb_62_col270;
        rel_values[65] = partial_ec_mul_output_limb_63_col271;
        rel_values[66] = partial_ec_mul_output_limb_64_col272;
        rel_values[67] = partial_ec_mul_output_limb_65_col273;
        rel_values[68] = partial_ec_mul_output_limb_66_col274;
        rel_values[69] = partial_ec_mul_output_limb_67_col275;
        rel_values[70] = partial_ec_mul_output_limb_68_col276;
        rel_values[71] = partial_ec_mul_output_limb_69_col277;
        rel_values[72] = partial_ec_mul_output_limb_70_col278;

        RelationEntry<73> ec_mul_entry_6(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{1, 0}, {0, 0}},  // positive multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_6);
    }

    // ===================== Pair 4, Lookup 7: chain_id_3, step=0, mult=-1 =====================
    // Input to fourth EC multiplication (ms_limb_high + output from previous step)
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_3;
        rel_values[1] = M31_0;  // step = 0
        rel_values[2] = M31_7340048;
        rel_values[3] = ms_limb_high_col58;

        // Zeros for limbs 4-16 (13 zeros, not 12!)
        for (int i = 4; i <= 16; i++) {
            rel_values[i] = M31_0;
        }

        // Limbs from col223-278 (partial_ec_mul_output_limb_15_col223 onwards)
        // Explicit assignments to avoid pointer arithmetic issues
        rel_values[17] = partial_ec_mul_output_limb_15_col223;
        rel_values[18] = partial_ec_mul_output_limb_16_col224;
        rel_values[19] = partial_ec_mul_output_limb_17_col225;
        rel_values[20] = partial_ec_mul_output_limb_18_col226;
        rel_values[21] = partial_ec_mul_output_limb_19_col227;
        rel_values[22] = partial_ec_mul_output_limb_20_col228;
        rel_values[23] = partial_ec_mul_output_limb_21_col229;
        rel_values[24] = partial_ec_mul_output_limb_22_col230;
        rel_values[25] = partial_ec_mul_output_limb_23_col231;
        rel_values[26] = partial_ec_mul_output_limb_24_col232;
        rel_values[27] = partial_ec_mul_output_limb_25_col233;
        rel_values[28] = partial_ec_mul_output_limb_26_col234;
        rel_values[29] = partial_ec_mul_output_limb_27_col235;
        rel_values[30] = partial_ec_mul_output_limb_28_col236;
        rel_values[31] = partial_ec_mul_output_limb_29_col237;
        rel_values[32] = partial_ec_mul_output_limb_30_col238;
        rel_values[33] = partial_ec_mul_output_limb_31_col239;
        rel_values[34] = partial_ec_mul_output_limb_32_col240;
        rel_values[35] = partial_ec_mul_output_limb_33_col241;
        rel_values[36] = partial_ec_mul_output_limb_34_col242;
        rel_values[37] = partial_ec_mul_output_limb_35_col243;
        rel_values[38] = partial_ec_mul_output_limb_36_col244;
        rel_values[39] = partial_ec_mul_output_limb_37_col245;
        rel_values[40] = partial_ec_mul_output_limb_38_col246;
        rel_values[41] = partial_ec_mul_output_limb_39_col247;
        rel_values[42] = partial_ec_mul_output_limb_40_col248;
        rel_values[43] = partial_ec_mul_output_limb_41_col249;
        rel_values[44] = partial_ec_mul_output_limb_42_col250;
        rel_values[45] = partial_ec_mul_output_limb_43_col251;
        rel_values[46] = partial_ec_mul_output_limb_44_col252;
        rel_values[47] = partial_ec_mul_output_limb_45_col253;
        rel_values[48] = partial_ec_mul_output_limb_46_col254;
        rel_values[49] = partial_ec_mul_output_limb_47_col255;
        rel_values[50] = partial_ec_mul_output_limb_48_col256;
        rel_values[51] = partial_ec_mul_output_limb_49_col257;
        rel_values[52] = partial_ec_mul_output_limb_50_col258;
        rel_values[53] = partial_ec_mul_output_limb_51_col259;
        rel_values[54] = partial_ec_mul_output_limb_52_col260;
        rel_values[55] = partial_ec_mul_output_limb_53_col261;
        rel_values[56] = partial_ec_mul_output_limb_54_col262;
        rel_values[57] = partial_ec_mul_output_limb_55_col263;
        rel_values[58] = partial_ec_mul_output_limb_56_col264;
        rel_values[59] = partial_ec_mul_output_limb_57_col265;
        rel_values[60] = partial_ec_mul_output_limb_58_col266;
        rel_values[61] = partial_ec_mul_output_limb_59_col267;
        rel_values[62] = partial_ec_mul_output_limb_60_col268;
        rel_values[63] = partial_ec_mul_output_limb_61_col269;
        rel_values[64] = partial_ec_mul_output_limb_62_col270;
        rel_values[65] = partial_ec_mul_output_limb_63_col271;
        rel_values[66] = partial_ec_mul_output_limb_64_col272;
        rel_values[67] = partial_ec_mul_output_limb_65_col273;
        rel_values[68] = partial_ec_mul_output_limb_66_col274;
        rel_values[69] = partial_ec_mul_output_limb_67_col275;
        rel_values[70] = partial_ec_mul_output_limb_68_col276;
        rel_values[71] = partial_ec_mul_output_limb_69_col277;
        rel_values[72] = partial_ec_mul_output_limb_70_col278;

        RelationEntry<73> ec_mul_entry_7(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{neg(m31(1)), 0}, {0, 0}},  // negative multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_7);
    }

    // ===================== Pair 4, Lookup 8: chain_id_3, step=1, mult=+1 =====================
    // Output from fourth EC multiplication at step 1
    {
        m31 rel_values[73];
        rel_values[0] = partial_ec_mul_chain_id_3;
        rel_values[1] = M31_1;  // step = 1
        rel_values[2] = partial_ec_mul_output_limb_0_col279;

        // Output limbs from columns 279-349
        // Explicit assignments to avoid pointer arithmetic issues
        rel_values[3] = partial_ec_mul_output_limb_1_col280;
        rel_values[4] = partial_ec_mul_output_limb_2_col281;
        rel_values[5] = partial_ec_mul_output_limb_3_col282;
        rel_values[6] = partial_ec_mul_output_limb_4_col283;
        rel_values[7] = partial_ec_mul_output_limb_5_col284;
        rel_values[8] = partial_ec_mul_output_limb_6_col285;
        rel_values[9] = partial_ec_mul_output_limb_7_col286;
        rel_values[10] = partial_ec_mul_output_limb_8_col287;
        rel_values[11] = partial_ec_mul_output_limb_9_col288;
        rel_values[12] = partial_ec_mul_output_limb_10_col289;
        rel_values[13] = partial_ec_mul_output_limb_11_col290;
        rel_values[14] = partial_ec_mul_output_limb_12_col291;
        rel_values[15] = partial_ec_mul_output_limb_13_col292;
        rel_values[16] = partial_ec_mul_output_limb_14_col293;
        rel_values[17] = partial_ec_mul_output_limb_15_col294;
        rel_values[18] = partial_ec_mul_output_limb_16_col295;
        rel_values[19] = partial_ec_mul_output_limb_17_col296;
        rel_values[20] = partial_ec_mul_output_limb_18_col297;
        rel_values[21] = partial_ec_mul_output_limb_19_col298;
        rel_values[22] = partial_ec_mul_output_limb_20_col299;
        rel_values[23] = partial_ec_mul_output_limb_21_col300;
        rel_values[24] = partial_ec_mul_output_limb_22_col301;
        rel_values[25] = partial_ec_mul_output_limb_23_col302;
        rel_values[26] = partial_ec_mul_output_limb_24_col303;
        rel_values[27] = partial_ec_mul_output_limb_25_col304;
        rel_values[28] = partial_ec_mul_output_limb_26_col305;
        rel_values[29] = partial_ec_mul_output_limb_27_col306;
        rel_values[30] = partial_ec_mul_output_limb_28_col307;
        rel_values[31] = partial_ec_mul_output_limb_29_col308;
        rel_values[32] = partial_ec_mul_output_limb_30_col309;
        rel_values[33] = partial_ec_mul_output_limb_31_col310;
        rel_values[34] = partial_ec_mul_output_limb_32_col311;
        rel_values[35] = partial_ec_mul_output_limb_33_col312;
        rel_values[36] = partial_ec_mul_output_limb_34_col313;
        rel_values[37] = partial_ec_mul_output_limb_35_col314;
        rel_values[38] = partial_ec_mul_output_limb_36_col315;
        rel_values[39] = partial_ec_mul_output_limb_37_col316;
        rel_values[40] = partial_ec_mul_output_limb_38_col317;
        rel_values[41] = partial_ec_mul_output_limb_39_col318;
        rel_values[42] = partial_ec_mul_output_limb_40_col319;
        rel_values[43] = partial_ec_mul_output_limb_41_col320;
        rel_values[44] = partial_ec_mul_output_limb_42_col321;
        rel_values[45] = partial_ec_mul_output_limb_43_col322;
        rel_values[46] = partial_ec_mul_output_limb_44_col323;
        rel_values[47] = partial_ec_mul_output_limb_45_col324;
        rel_values[48] = partial_ec_mul_output_limb_46_col325;
        rel_values[49] = partial_ec_mul_output_limb_47_col326;
        rel_values[50] = partial_ec_mul_output_limb_48_col327;
        rel_values[51] = partial_ec_mul_output_limb_49_col328;
        rel_values[52] = partial_ec_mul_output_limb_50_col329;
        rel_values[53] = partial_ec_mul_output_limb_51_col330;
        rel_values[54] = partial_ec_mul_output_limb_52_col331;
        rel_values[55] = partial_ec_mul_output_limb_53_col332;
        rel_values[56] = partial_ec_mul_output_limb_54_col333;
        rel_values[57] = partial_ec_mul_output_limb_55_col334;
        rel_values[58] = partial_ec_mul_output_limb_56_col335;
        rel_values[59] = partial_ec_mul_output_limb_57_col336;
        rel_values[60] = partial_ec_mul_output_limb_58_col337;
        rel_values[61] = partial_ec_mul_output_limb_59_col338;
        rel_values[62] = partial_ec_mul_output_limb_60_col339;
        rel_values[63] = partial_ec_mul_output_limb_61_col340;
        rel_values[64] = partial_ec_mul_output_limb_62_col341;
        rel_values[65] = partial_ec_mul_output_limb_63_col342;
        rel_values[66] = partial_ec_mul_output_limb_64_col343;
        rel_values[67] = partial_ec_mul_output_limb_65_col344;
        rel_values[68] = partial_ec_mul_output_limb_66_col345;
        rel_values[69] = partial_ec_mul_output_limb_67_col346;
        rel_values[70] = partial_ec_mul_output_limb_68_col347;
        rel_values[71] = partial_ec_mul_output_limb_69_col348;
        rel_values[72] = partial_ec_mul_output_limb_70_col349;

        RelationEntry<73> ec_mul_entry_8(
            pedersen_eval->partial_ec_mul_lookup_elements,
            qm31{{1, 0}, {0, 0}},  // positive multiplicity
            rel_values
        );
        cuda_evaluator1.add_to_relation<73>(ec_mul_entry_8);
    }

    // ===================== MemVerify: Write result to memory =====================
    // Verifies that the final result is written to memory at address (instance_addr + 2)
    // Uses only the first 28 limbs (columns 294-321) of the fourth EC mul output
    // IMPORTANT: Address must be the FIRST parameter to match CPU implementation!
    mem_verify_evaluate<EvaluatorT>(
        add(instance_addr, M31_2),  // Address FIRST (not last!)
        partial_ec_mul_output_limb_15_col294,
        partial_ec_mul_output_limb_16_col295,
        partial_ec_mul_output_limb_17_col296,
        partial_ec_mul_output_limb_18_col297,
        partial_ec_mul_output_limb_19_col298,
        partial_ec_mul_output_limb_20_col299,
        partial_ec_mul_output_limb_21_col300,
        partial_ec_mul_output_limb_22_col301,
        partial_ec_mul_output_limb_23_col302,
        partial_ec_mul_output_limb_24_col303,
        partial_ec_mul_output_limb_25_col304,
        partial_ec_mul_output_limb_26_col305,
        partial_ec_mul_output_limb_27_col306,
        partial_ec_mul_output_limb_28_col307,
        partial_ec_mul_output_limb_29_col308,
        partial_ec_mul_output_limb_30_col309,
        partial_ec_mul_output_limb_31_col310,
        partial_ec_mul_output_limb_32_col311,
        partial_ec_mul_output_limb_33_col312,
        partial_ec_mul_output_limb_34_col313,
        partial_ec_mul_output_limb_35_col314,
        partial_ec_mul_output_limb_36_col315,
        partial_ec_mul_output_limb_37_col316,
        partial_ec_mul_output_limb_38_col317,
        partial_ec_mul_output_limb_39_col318,
        partial_ec_mul_output_limb_40_col319,
        partial_ec_mul_output_limb_41_col320,
        partial_ec_mul_output_limb_42_col321,
        pedersen_result_id_col350,
        pedersen_eval->memory_address_to_id_lookup_elements,
        pedersen_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== Complete constraint evaluation =====================
    // Save constraint index and accumulated result for this row
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// Pedersen Main Function: orchestrates the entire evaluation process
// ============================================================================

extern "C"
void evaluate_pedersen_builtin(
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

    PedersenBuiltin_Eval *pedersen_eval =
        (PedersenBuiltin_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // Clone trace0 (preprocessed) and trace1 (base trace) separately
    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PedersenBuiltin_Eval *device_pedersen_eval =
        cuda_malloc<PedersenBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<PedersenBuiltin_Eval>(
        pedersen_eval, device_pedersen_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_pedersen_builtin");

    int block_dim = eval_domain_size < PEDERSEN_BUILTIN_THREAD_COUNT_MAX
        ? eval_domain_size
        : PEDERSEN_BUILTIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_pedersen_builtin_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_pedersen_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_pedersen_builtin_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_pedersen_eval,
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
    global_timer.end("evaluate_pedersen_builtin");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_pedersen_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
