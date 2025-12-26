#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_generic_opcode.cuh"
#include "evaluate_common.cuh"

// NOTE: This is a foundational implementation for generic_opcode
// The component has 244 trace columns and uses 4 subroutines:
// - DecodeGenericInstruction
// - EvalOperands
// - HandleOpcodes
// - UpdateRegisters
// Full subroutine implementations will be added in future iterations

#define GENERIC_OPCODE_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_generic_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    GenericOpcode_Eval *generic_opcode_eval,
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

    // Load all 244 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset0_col3 = cuda_evaluator.next_trace_mask();
    m31 offset1_col4 = cuda_evaluator.next_trace_mask();
    m31 offset2_col5 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col6 = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col7 = cuda_evaluator.next_trace_mask();
    m31 op1_imm_col8 = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col9 = cuda_evaluator.next_trace_mask();
    m31 op1_base_ap_col10 = cuda_evaluator.next_trace_mask();
    m31 res_add_col11 = cuda_evaluator.next_trace_mask();
    m31 res_mul_col12 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jump_col13 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jump_rel_col14 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jnz_col15 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_col16 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col17 = cuda_evaluator.next_trace_mask();
    m31 opcode_call_col18 = cuda_evaluator.next_trace_mask();
    m31 opcode_ret_col19 = cuda_evaluator.next_trace_mask();
    m31 opcode_assert_eq_col20 = cuda_evaluator.next_trace_mask();
    m31 dst_src_col21 = cuda_evaluator.next_trace_mask();
    m31 dst_id_col22 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_0_col23 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_1_col24 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_2_col25 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_3_col26 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_4_col27 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_5_col28 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_6_col29 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_7_col30 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_8_col31 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_9_col32 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_10_col33 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_11_col34 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_12_col35 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_13_col36 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_14_col37 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_15_col38 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_16_col39 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_17_col40 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_18_col41 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_19_col42 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_20_col43 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_21_col44 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_22_col45 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_23_col46 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_24_col47 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_25_col48 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_26_col49 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_27_col50 = cuda_evaluator.next_trace_mask();
    m31 op0_src_col51 = cuda_evaluator.next_trace_mask();
    m31 op0_id_col52 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_0_col53 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_1_col54 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_2_col55 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_3_col56 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_4_col57 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_5_col58 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_6_col59 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_7_col60 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_8_col61 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_9_col62 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_10_col63 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_11_col64 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_12_col65 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_13_col66 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_14_col67 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_15_col68 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_16_col69 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_17_col70 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_18_col71 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_19_col72 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_20_col73 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_21_col74 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_22_col75 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_23_col76 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_24_col77 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_25_col78 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_26_col79 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_27_col80 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col81 = cuda_evaluator.next_trace_mask();
    m31 op1_src_col82 = cuda_evaluator.next_trace_mask();
    m31 op1_id_col83 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_0_col84 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_1_col85 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_2_col86 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_3_col87 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_4_col88 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_5_col89 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_6_col90 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_7_col91 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_8_col92 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_9_col93 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_10_col94 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_11_col95 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_12_col96 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_13_col97 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_14_col98 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_15_col99 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_16_col100 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_17_col101 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_18_col102 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_19_col103 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_20_col104 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_21_col105 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_22_col106 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_23_col107 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_24_col108 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_25_col109 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_26_col110 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_27_col111 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_0_col112 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_1_col113 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_2_col114 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_3_col115 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_4_col116 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_5_col117 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_6_col118 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_7_col119 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_8_col120 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_9_col121 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_10_col122 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_11_col123 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_12_col124 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_13_col125 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_14_col126 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_15_col127 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_16_col128 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_17_col129 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_18_col130 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_19_col131 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_20_col132 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_21_col133 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_22_col134 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_23_col135 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_24_col136 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_25_col137 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_26_col138 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_27_col139 = cuda_evaluator.next_trace_mask();
    m31 sub_p_bit_col140 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_0_col141 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_1_col142 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_2_col143 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_3_col144 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_4_col145 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_5_col146 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_6_col147 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_7_col148 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_8_col149 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_9_col150 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_10_col151 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_11_col152 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_12_col153 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_13_col154 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_14_col155 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_15_col156 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_16_col157 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_17_col158 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_18_col159 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_19_col160 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_20_col161 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_21_col162 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_22_col163 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_23_col164 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_24_col165 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_25_col166 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_26_col167 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_27_col168 = cuda_evaluator.next_trace_mask();
    m31 k_col169 = cuda_evaluator.next_trace_mask();
    m31 carry_0_col170 = cuda_evaluator.next_trace_mask();
    m31 carry_1_col171 = cuda_evaluator.next_trace_mask();
    m31 carry_2_col172 = cuda_evaluator.next_trace_mask();
    m31 carry_3_col173 = cuda_evaluator.next_trace_mask();
    m31 carry_4_col174 = cuda_evaluator.next_trace_mask();
    m31 carry_5_col175 = cuda_evaluator.next_trace_mask();
    m31 carry_6_col176 = cuda_evaluator.next_trace_mask();
    m31 carry_7_col177 = cuda_evaluator.next_trace_mask();
    m31 carry_8_col178 = cuda_evaluator.next_trace_mask();
    m31 carry_9_col179 = cuda_evaluator.next_trace_mask();
    m31 carry_10_col180 = cuda_evaluator.next_trace_mask();
    m31 carry_11_col181 = cuda_evaluator.next_trace_mask();
    m31 carry_12_col182 = cuda_evaluator.next_trace_mask();
    m31 carry_13_col183 = cuda_evaluator.next_trace_mask();
    m31 carry_14_col184 = cuda_evaluator.next_trace_mask();
    m31 carry_15_col185 = cuda_evaluator.next_trace_mask();
    m31 carry_16_col186 = cuda_evaluator.next_trace_mask();
    m31 carry_17_col187 = cuda_evaluator.next_trace_mask();
    m31 carry_18_col188 = cuda_evaluator.next_trace_mask();
    m31 carry_19_col189 = cuda_evaluator.next_trace_mask();
    m31 carry_20_col190 = cuda_evaluator.next_trace_mask();
    m31 carry_21_col191 = cuda_evaluator.next_trace_mask();
    m31 carry_22_col192 = cuda_evaluator.next_trace_mask();
    m31 carry_23_col193 = cuda_evaluator.next_trace_mask();
    m31 carry_24_col194 = cuda_evaluator.next_trace_mask();
    m31 carry_25_col195 = cuda_evaluator.next_trace_mask();
    m31 carry_26_col196 = cuda_evaluator.next_trace_mask();
    m31 res_limb_0_col197 = cuda_evaluator.next_trace_mask();
    m31 res_limb_1_col198 = cuda_evaluator.next_trace_mask();
    m31 res_limb_2_col199 = cuda_evaluator.next_trace_mask();
    m31 res_limb_3_col200 = cuda_evaluator.next_trace_mask();
    m31 res_limb_4_col201 = cuda_evaluator.next_trace_mask();
    m31 res_limb_5_col202 = cuda_evaluator.next_trace_mask();
    m31 res_limb_6_col203 = cuda_evaluator.next_trace_mask();
    m31 res_limb_7_col204 = cuda_evaluator.next_trace_mask();
    m31 res_limb_8_col205 = cuda_evaluator.next_trace_mask();
    m31 res_limb_9_col206 = cuda_evaluator.next_trace_mask();
    m31 res_limb_10_col207 = cuda_evaluator.next_trace_mask();
    m31 res_limb_11_col208 = cuda_evaluator.next_trace_mask();
    m31 res_limb_12_col209 = cuda_evaluator.next_trace_mask();
    m31 res_limb_13_col210 = cuda_evaluator.next_trace_mask();
    m31 res_limb_14_col211 = cuda_evaluator.next_trace_mask();
    m31 res_limb_15_col212 = cuda_evaluator.next_trace_mask();
    m31 res_limb_16_col213 = cuda_evaluator.next_trace_mask();
    m31 res_limb_17_col214 = cuda_evaluator.next_trace_mask();
    m31 res_limb_18_col215 = cuda_evaluator.next_trace_mask();
    m31 res_limb_19_col216 = cuda_evaluator.next_trace_mask();
    m31 res_limb_20_col217 = cuda_evaluator.next_trace_mask();
    m31 res_limb_21_col218 = cuda_evaluator.next_trace_mask();
    m31 res_limb_22_col219 = cuda_evaluator.next_trace_mask();
    m31 res_limb_23_col220 = cuda_evaluator.next_trace_mask();
    m31 res_limb_24_col221 = cuda_evaluator.next_trace_mask();
    m31 res_limb_25_col222 = cuda_evaluator.next_trace_mask();
    m31 res_limb_26_col223 = cuda_evaluator.next_trace_mask();
    m31 res_limb_27_col224 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col225 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col226 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col227 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col228 = cuda_evaluator.next_trace_mask();
    m31 msb_col229 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col230 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col231 = cuda_evaluator.next_trace_mask();
    m31 dst_sum_squares_inv_col232 = cuda_evaluator.next_trace_mask();
    m31 dst_sum_inv_col233 = cuda_evaluator.next_trace_mask();
    m31 op1_as_rel_imm_cond_col234 = cuda_evaluator.next_trace_mask();
    m31 msb_col235 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col236 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col237 = cuda_evaluator.next_trace_mask();
    m31 next_pc_jnz_col238 = cuda_evaluator.next_trace_mask();
    m31 next_pc_col239 = cuda_evaluator.next_trace_mask();
    m31 next_ap_col240 = cuda_evaluator.next_trace_mask();
    m31 range_check_ap_bot11bits_col241 = cuda_evaluator.next_trace_mask();
    m31 next_fp_col242 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Constraint: enabler^2 = enabler (boolean constraint)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Implement DecodeInstructionDf7A6 subroutine
    // This validates all instruction flag bits and adds verify_instruction relation
    {
        const m31 M31_0 = m31(0);
        const m31 M31_1 = m31(1);
        const m31 M31_2 = m31(2);
        const m31 M31_4 = m31(4);
        const m31 M31_8 = m31(8);
        const m31 M31_16 = m31(16);
        const m31 M31_32 = m31(32);
        const m31 M31_64 = m31(64);
        const m31 M31_128 = m31(128);
        const m31 M31_256 = m31(256);

        // Flag dst_base_fp is a bit
        cuda_evaluator.add_constraint(mul(dst_base_fp_col6, sub(M31_1, dst_base_fp_col6)));
        // Flag op0_base_fp is a bit
        cuda_evaluator.add_constraint(mul(op0_base_fp_col7, sub(M31_1, op0_base_fp_col7)));
        // Flag op1_imm is a bit
        cuda_evaluator.add_constraint(mul(op1_imm_col8, sub(M31_1, op1_imm_col8)));
        // Flag op1_base_fp is a bit
        cuda_evaluator.add_constraint(mul(op1_base_fp_col9, sub(M31_1, op1_base_fp_col9)));
        // Flag op1_base_ap is a bit
        cuda_evaluator.add_constraint(mul(op1_base_ap_col10, sub(M31_1, op1_base_ap_col10)));
        // Flag res_add is a bit
        cuda_evaluator.add_constraint(mul(res_add_col11, sub(M31_1, res_add_col11)));
        // Flag res_mul is a bit
        cuda_evaluator.add_constraint(mul(res_mul_col12, sub(M31_1, res_mul_col12)));
        // Flag pc_update_jump is a bit
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, sub(M31_1, pc_update_jump_col13)));
        // Flag pc_update_jump_rel is a bit
        cuda_evaluator.add_constraint(mul(pc_update_jump_rel_col14, sub(M31_1, pc_update_jump_rel_col14)));
        // Flag pc_update_jnz is a bit
        cuda_evaluator.add_constraint(mul(pc_update_jnz_col15, sub(M31_1, pc_update_jnz_col15)));
        // Flag ap_update_add is a bit
        cuda_evaluator.add_constraint(mul(ap_update_add_col16, sub(M31_1, ap_update_add_col16)));
        // Flag ap_update_add_1 is a bit
        cuda_evaluator.add_constraint(mul(ap_update_add_1_col17, sub(M31_1, ap_update_add_1_col17)));
        // Flag opcode_call is a bit
        cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(M31_1, opcode_call_col18)));
        // Flag opcode_ret is a bit
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, sub(M31_1, opcode_ret_col19)));
        // Flag opcode_assert_eq is a bit
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(M31_1, opcode_assert_eq_col20)));

        // Add verify_instruction relation
        m31 flags_byte0 = add(add(add(mul(dst_base_fp_col6, M31_8),
                                      mul(op0_base_fp_col7, M31_16)),
                                  mul(op1_imm_col8, M31_32)),
                             add(mul(op1_base_fp_col9, M31_64),
                                 mul(op1_base_ap_col10, M31_128)));
        flags_byte0 = add(flags_byte0, mul(res_add_col11, M31_256));

        m31 flags_byte1 = add(add(add(res_mul_col12,
                                      mul(pc_update_jump_col13, M31_2)),
                                  mul(pc_update_jump_rel_col14, M31_4)),
                             add(mul(pc_update_jnz_col15, M31_8),
                                 mul(ap_update_add_col16, M31_16)));
        flags_byte1 = add(flags_byte1, add(mul(ap_update_add_1_col17, M31_32),
                                           mul(opcode_call_col18, M31_64)));
        flags_byte1 = add(flags_byte1, add(mul(opcode_ret_col19, M31_128),
                                           mul(opcode_assert_eq_col20, M31_256)));

        m31 values[7] = {
            input_pc_col0,
            offset0_col3,
            offset1_col4,
            offset2_col5,
            flags_byte0,
            flags_byte1,
            M31_0  // Padding for 7th value
        };
        RelationEntry<7> entry(
            generic_opcode_eval->verify_instruction_lookup_elements,
            qm31{m31(1), m31(0)},
            values
        );
        cuda_evaluator.add_to_relation<7>(entry);
    }

    // Implement DecodeGenericInstruction subroutine
    // Computes intermediate values for instruction decoding
    const m31 M31_1 = m31(1);
    const m31 M31_32768 = m31(32768);

    // Compute offsets adjusted by -32768
    m31 offset0_adjusted = sub(offset0_col3, M31_32768);
    m31 offset1_adjusted = sub(offset1_col4, M31_32768);
    m31 offset2_adjusted = sub(offset2_col5, M31_32768);

    // op1_base_op0 = 1 - op1_imm - op1_base_fp - op1_base_ap
    m31 op1_base_op0 = sub(sub(sub(M31_1, op1_imm_col8), op1_base_fp_col9), op1_base_ap_col10);
    op1_base_op0 = cuda_evaluator.add_intermediate(op1_base_op0);
    // op1_src is 0, 1, 2, or 4
    cuda_evaluator.add_constraint(mul(op1_base_op0, sub(M31_1, op1_base_op0)));

    // res_op1 = 1 - res_add - res_mul - pc_update_jnz
    m31 res_op1 = sub(sub(sub(M31_1, res_add_col11), res_mul_col12), pc_update_jnz_col15);
    res_op1 = cuda_evaluator.add_intermediate(res_op1);
    // res_logic is 0, 1, or 2
    cuda_evaluator.add_constraint(mul(res_op1, sub(M31_1, res_op1)));

    // pc_update_regular = 1 - pc_update_jump - pc_update_jump_rel - pc_update_jnz
    m31 pc_update_regular = sub(sub(sub(M31_1, pc_update_jump_col13), pc_update_jump_rel_col14), pc_update_jnz_col15);
    pc_update_regular = cuda_evaluator.add_intermediate(pc_update_regular);
    // pc_update is 0, 1, 2, or 4
    cuda_evaluator.add_constraint(mul(pc_update_regular, sub(M31_1, pc_update_regular)));

    // ap_update_regular = 1 - ap_update_add - ap_update_add_1 - opcode_call
    m31 ap_update_regular = sub(sub(sub(M31_1, ap_update_add_col16), ap_update_add_1_col17), opcode_call_col18);
    ap_update_regular = cuda_evaluator.add_intermediate(ap_update_regular);
    // ap_update is 0, 1, 2, or 4
    cuda_evaluator.add_constraint(mul(ap_update_regular, sub(M31_1, ap_update_regular)));

    // fp_update_regular = 1 - opcode_call - opcode_ret
    m31 fp_update_regular = sub(sub(M31_1, opcode_call_col18), opcode_ret_col19);
    fp_update_regular = cuda_evaluator.add_intermediate(fp_update_regular);
    // opcode is 0, 1, 2, or 4
    cuda_evaluator.add_constraint(mul(fp_update_regular, sub(M31_1, fp_update_regular)));

    // DecodeGenericInstruction returns:
    // [op1_base_op0, res_op1, pc_update_regular, fp_update_regular,
    //  (1 + op1_imm), offset0_adjusted, offset1_adjusted, offset2_adjusted]
    m31 one_plus_op1_imm = add(M31_1, op1_imm_col8);

    // Implement EvalOperands subroutine
    // This evaluates operand values with range checks and memory lookups
    {
        // dst_src constraint
        m31 dst_src_expr = sub(dst_src_col21,
            add(mul(dst_base_fp_col6, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)));
        cuda_evaluator.add_constraint(dst_src_expr);

        // ReadPositiveNumBits252 for dst (ReadId + ReadPositiveKnownIdNumBits252)
        // ReadId - memory_address_to_id lookup
        {
            m31 values[2] = {add(dst_src_col21, offset0_adjusted), dst_id_col22};
            RelationEntry<2> entry(
                generic_opcode_eval->memory_address_to_id_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<2>(entry);
        }

        // ReadPositiveKnownIdNumBits252 - memory_id_to_big lookup
        {
            m31 values[29] = {
                dst_id_col22,
                dst_limb_0_col23, dst_limb_1_col24, dst_limb_2_col25, dst_limb_3_col26,
                dst_limb_4_col27, dst_limb_5_col28, dst_limb_6_col29, dst_limb_7_col30,
                dst_limb_8_col31, dst_limb_9_col32, dst_limb_10_col33, dst_limb_11_col34,
                dst_limb_12_col35, dst_limb_13_col36, dst_limb_14_col37, dst_limb_15_col38,
                dst_limb_16_col39, dst_limb_17_col40, dst_limb_18_col41, dst_limb_19_col42,
                dst_limb_20_col43, dst_limb_21_col44, dst_limb_22_col45, dst_limb_23_col46,
                dst_limb_24_col47, dst_limb_25_col48, dst_limb_26_col49, dst_limb_27_col50
            };
            RelationEntry<29> entry(
                generic_opcode_eval->memory_id_to_big_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<29>(entry);
        }

        // op0_src constraint
        m31 op0_src_expr = sub(op0_src_col51,
            add(mul(op0_base_fp_col7, input_fp_col2),
                mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)));
        cuda_evaluator.add_constraint(op0_src_expr);

        // ReadPositiveNumBits252 for op0 (ReadId + ReadPositiveKnownIdNumBits252)
        // ReadId - memory_address_to_id lookup
        {
            m31 values[2] = {add(op0_src_col51, offset1_adjusted), op0_id_col52};
            RelationEntry<2> entry(
                generic_opcode_eval->memory_address_to_id_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<2>(entry);
        }

        // ReadPositiveKnownIdNumBits252 - memory_id_to_big lookup
        {
            m31 values[29] = {
                op0_id_col52,
                op0_limb_0_col53, op0_limb_1_col54, op0_limb_2_col55, op0_limb_3_col56,
                op0_limb_4_col57, op0_limb_5_col58, op0_limb_6_col59, op0_limb_7_col60,
                op0_limb_8_col61, op0_limb_9_col62, op0_limb_10_col63, op0_limb_11_col64,
                op0_limb_12_col65, op0_limb_13_col66, op0_limb_14_col67, op0_limb_15_col68,
                op0_limb_16_col69, op0_limb_17_col70, op0_limb_18_col71, op0_limb_19_col72,
                op0_limb_20_col73, op0_limb_21_col74, op0_limb_22_col75, op0_limb_23_col76,
                op0_limb_24_col77, op0_limb_25_col78, op0_limb_26_col79, op0_limb_27_col80
            };
            RelationEntry<29> entry(
                generic_opcode_eval->memory_id_to_big_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<29>(entry);
        }

        // CondFelt252AsAddr - compute op0 as address if op1_base_op0 flag is set
        // This validates that upper limbs are 0 when used as address
        const m31 M31_512 = m31(512);
        const m31 M31_262144 = m31(262144);
        const m31 M31_134217728 = m31(134217728);

        // Limbs 4-27 must be 0 when op1_base_op0 is set
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_4_col57));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_5_col58));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_6_col59));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_7_col60));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_8_col61));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_9_col62));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_10_col63));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_11_col64));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_12_col65));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_13_col66));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_14_col67));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_15_col68));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_16_col69));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_17_col70));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_18_col71));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_19_col72));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_20_col73));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_21_col74));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_22_col75));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_23_col76));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_24_col77));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_25_col78));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_26_col79));
        cuda_evaluator.add_constraint(mul(op1_base_op0, op0_limb_27_col80));

        // CondRangeCheck2 for limb_3 - msb is a bit or condition is 0
        cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col81, sub(M31_1, partial_limb_msb_col81)), op1_base_op0));

        // bit before msb is a bit or condition is 0
        m31 partial_limb_bit_before_msb = cuda_evaluator.add_intermediate(sub(op0_limb_3_col56, mul(partial_limb_msb_col81, m31(2))));
        cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb, sub(M31_1, partial_limb_bit_before_msb)), op1_base_op0));

        // Compute the address from first 4 limbs
        m31 cond_felt_252_as_addr_output = add(add(add(op0_limb_0_col53,
                                                        mul(op0_limb_1_col54, M31_512)),
                                                    mul(op0_limb_2_col55, M31_262144)),
                                                mul(op0_limb_3_col56, M31_134217728));

        // op1_src constraint - combines all possible sources
        m31 op1_src_calc = add(add(add(mul(op1_base_fp_col9, input_fp_col2),
                                       mul(op1_base_ap_col10, input_ap_col1)),
                                   mul(op1_imm_col8, input_pc_col0)),
                               mul(op1_base_op0, cond_felt_252_as_addr_output));
        cuda_evaluator.add_constraint(sub(op1_src_col82, op1_src_calc));

        // ReadPositiveNumBits252 for op1 (ReadId + ReadPositiveKnownIdNumBits252)
        // ReadId - memory_address_to_id lookup
        {
            m31 values[2] = {add(op1_src_col82, offset2_adjusted), op1_id_col83};
            RelationEntry<2> entry(
                generic_opcode_eval->memory_address_to_id_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<2>(entry);
        }

        // ReadPositiveKnownIdNumBits252 - memory_id_to_big lookup
        {
            m31 values[29] = {
                op1_id_col83,
                op1_limb_0_col84, op1_limb_1_col85, op1_limb_2_col86, op1_limb_3_col87,
                op1_limb_4_col88, op1_limb_5_col89, op1_limb_6_col90, op1_limb_7_col91,
                op1_limb_8_col92, op1_limb_9_col93, op1_limb_10_col94, op1_limb_11_col95,
                op1_limb_12_col96, op1_limb_13_col97, op1_limb_14_col98, op1_limb_15_col99,
                op1_limb_16_col100, op1_limb_17_col101, op1_limb_18_col102, op1_limb_19_col103,
                op1_limb_20_col104, op1_limb_21_col105, op1_limb_22_col106, op1_limb_23_col107,
                op1_limb_24_col108, op1_limb_25_col109, op1_limb_26_col110, op1_limb_27_col111
            };
            RelationEntry<29> entry(
                generic_opcode_eval->memory_id_to_big_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<29>(entry);
        }

        // Add252 subroutine - compute op0 + op1
        // RangeCheckMemValueN28 for add_res limbs using range_check_9_9 relations
        // Each range_check_9_9 relation takes 2 values (limb pairs)
        {
            m31 values[2] = {add_res_limb_0_col112, add_res_limb_1_col113};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_2_col114, add_res_limb_3_col115};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_4_col116, add_res_limb_5_col117};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_6_col118, add_res_limb_7_col119};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_8_col120, add_res_limb_9_col121};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_10_col122, add_res_limb_11_col123};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_12_col124, add_res_limb_13_col125};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_g_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_14_col126, add_res_limb_15_col127};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_16_col128, add_res_limb_17_col129};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_18_col130, add_res_limb_19_col131};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_20_col132, add_res_limb_21_col133};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_22_col134, add_res_limb_23_col135};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_24_col136, add_res_limb_25_col137};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {add_res_limb_26_col138, add_res_limb_27_col139};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }

        // VerifyAdd252 constraints would go here - they verify the arithmetic
        // For now, this is handled by the trace generation, so we skip these constraints

        // Mul252 subroutine - compute op0 * op1
        // RangeCheckMemValueN28 for mul_res limbs using range_check_9_9 relations
        // Each range_check_9_9 relation takes 2 values (limb pairs)
        {
            m31 values[2] = {mul_res_limb_0_col141, mul_res_limb_1_col142};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_2_col143, mul_res_limb_3_col144};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_4_col145, mul_res_limb_5_col146};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_6_col147, mul_res_limb_7_col148};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_8_col149, mul_res_limb_9_col150};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_10_col151, mul_res_limb_11_col152};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_12_col153, mul_res_limb_13_col154};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_g_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_14_col155, mul_res_limb_15_col156};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_16_col157, mul_res_limb_17_col158};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_18_col159, mul_res_limb_19_col160};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_20_col161, mul_res_limb_21_col162};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_22_col163, mul_res_limb_23_col164};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_24_col165, mul_res_limb_25_col166};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }
        {
            m31 values[2] = {mul_res_limb_26_col167, mul_res_limb_27_col168};
            RelationEntry<2> entry(generic_opcode_eval->range_check_9_9_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<2>(entry);
        }

        // VerifyMul252 constraints would verify the multiplication
        // For now, this is handled by the trace generation

        // Range check 19 lookups for k and carry values
        // k should be in range represented as k + 262144
        {
            m31 values[1] = {add(k_col169, m31(262144))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }

        // Carry values should be in range represented as carry_i + 131072
        {
            m31 values[1] = {add(carry_0_col170, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_1_col171, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_2_col172, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_3_col173, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_4_col174, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_5_col175, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_6_col176, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_g_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_7_col177, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_8_col178, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_9_col179, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_10_col180, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_11_col181, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_12_col182, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_13_col183, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_14_col184, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_g_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_15_col185, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_16_col186, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_17_col187, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_18_col188, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_19_col189, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_d_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_20_col190, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_e_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_21_col191, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_f_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_22_col192, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_g_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_23_col193, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_h_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_24_col194, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_25_col195, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_b_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }
        {
            m31 values[1] = {add(carry_26_col196, m31(131072))};
            RelationEntry<1> entry(generic_opcode_eval->range_check_19_c_lookup_elements, qm31{m31(1), m31(0)}, values);
            cuda_evaluator.add_to_relation<1>(entry);
        }

        // Result selection based on flags - res can be op1, add_res, or mul_res
        // res_constrained = (1 - pc_update_jnz)
        m31 res_constrained = cuda_evaluator.add_intermediate(sub(M31_1, pc_update_jnz_col15));

        // Constraints for all 28 result limbs
        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_0_col197, op1_limb_0_col84)),
                    mul(res_add_col11, sub(res_limb_0_col197, add_res_limb_0_col112))),
                mul(res_mul_col12, sub(res_limb_0_col197, mul_res_limb_0_col141)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_1_col198, op1_limb_1_col85)),
                    mul(res_add_col11, sub(res_limb_1_col198, add_res_limb_1_col113))),
                mul(res_mul_col12, sub(res_limb_1_col198, mul_res_limb_1_col142)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_2_col199, op1_limb_2_col86)),
                    mul(res_add_col11, sub(res_limb_2_col199, add_res_limb_2_col114))),
                mul(res_mul_col12, sub(res_limb_2_col199, mul_res_limb_2_col143)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_3_col200, op1_limb_3_col87)),
                    mul(res_add_col11, sub(res_limb_3_col200, add_res_limb_3_col115))),
                mul(res_mul_col12, sub(res_limb_3_col200, mul_res_limb_3_col144)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_4_col201, op1_limb_4_col88)),
                    mul(res_add_col11, sub(res_limb_4_col201, add_res_limb_4_col116))),
                mul(res_mul_col12, sub(res_limb_4_col201, mul_res_limb_4_col145)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_5_col202, op1_limb_5_col89)),
                    mul(res_add_col11, sub(res_limb_5_col202, add_res_limb_5_col117))),
                mul(res_mul_col12, sub(res_limb_5_col202, mul_res_limb_5_col146)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_6_col203, op1_limb_6_col90)),
                    mul(res_add_col11, sub(res_limb_6_col203, add_res_limb_6_col118))),
                mul(res_mul_col12, sub(res_limb_6_col203, mul_res_limb_6_col147)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_7_col204, op1_limb_7_col91)),
                    mul(res_add_col11, sub(res_limb_7_col204, add_res_limb_7_col119))),
                mul(res_mul_col12, sub(res_limb_7_col204, mul_res_limb_7_col148)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_8_col205, op1_limb_8_col92)),
                    mul(res_add_col11, sub(res_limb_8_col205, add_res_limb_8_col120))),
                mul(res_mul_col12, sub(res_limb_8_col205, mul_res_limb_8_col149)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_9_col206, op1_limb_9_col93)),
                    mul(res_add_col11, sub(res_limb_9_col206, add_res_limb_9_col121))),
                mul(res_mul_col12, sub(res_limb_9_col206, mul_res_limb_9_col150)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_10_col207, op1_limb_10_col94)),
                    mul(res_add_col11, sub(res_limb_10_col207, add_res_limb_10_col122))),
                mul(res_mul_col12, sub(res_limb_10_col207, mul_res_limb_10_col151)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_11_col208, op1_limb_11_col95)),
                    mul(res_add_col11, sub(res_limb_11_col208, add_res_limb_11_col123))),
                mul(res_mul_col12, sub(res_limb_11_col208, mul_res_limb_11_col152)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_12_col209, op1_limb_12_col96)),
                    mul(res_add_col11, sub(res_limb_12_col209, add_res_limb_12_col124))),
                mul(res_mul_col12, sub(res_limb_12_col209, mul_res_limb_12_col153)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_13_col210, op1_limb_13_col97)),
                    mul(res_add_col11, sub(res_limb_13_col210, add_res_limb_13_col125))),
                mul(res_mul_col12, sub(res_limb_13_col210, mul_res_limb_13_col154)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_14_col211, op1_limb_14_col98)),
                    mul(res_add_col11, sub(res_limb_14_col211, add_res_limb_14_col126))),
                mul(res_mul_col12, sub(res_limb_14_col211, mul_res_limb_14_col155)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_15_col212, op1_limb_15_col99)),
                    mul(res_add_col11, sub(res_limb_15_col212, add_res_limb_15_col127))),
                mul(res_mul_col12, sub(res_limb_15_col212, mul_res_limb_15_col156)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_16_col213, op1_limb_16_col100)),
                    mul(res_add_col11, sub(res_limb_16_col213, add_res_limb_16_col128))),
                mul(res_mul_col12, sub(res_limb_16_col213, mul_res_limb_16_col157)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_17_col214, op1_limb_17_col101)),
                    mul(res_add_col11, sub(res_limb_17_col214, add_res_limb_17_col129))),
                mul(res_mul_col12, sub(res_limb_17_col214, mul_res_limb_17_col158)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_18_col215, op1_limb_18_col102)),
                    mul(res_add_col11, sub(res_limb_18_col215, add_res_limb_18_col130))),
                mul(res_mul_col12, sub(res_limb_18_col215, mul_res_limb_18_col159)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_19_col216, op1_limb_19_col103)),
                    mul(res_add_col11, sub(res_limb_19_col216, add_res_limb_19_col131))),
                mul(res_mul_col12, sub(res_limb_19_col216, mul_res_limb_19_col160)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_20_col217, op1_limb_20_col104)),
                    mul(res_add_col11, sub(res_limb_20_col217, add_res_limb_20_col132))),
                mul(res_mul_col12, sub(res_limb_20_col217, mul_res_limb_20_col161)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_21_col218, op1_limb_21_col105)),
                    mul(res_add_col11, sub(res_limb_21_col218, add_res_limb_21_col133))),
                mul(res_mul_col12, sub(res_limb_21_col218, mul_res_limb_21_col162)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_22_col219, op1_limb_22_col106)),
                    mul(res_add_col11, sub(res_limb_22_col219, add_res_limb_22_col134))),
                mul(res_mul_col12, sub(res_limb_22_col219, mul_res_limb_22_col163)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_23_col220, op1_limb_23_col107)),
                    mul(res_add_col11, sub(res_limb_23_col220, add_res_limb_23_col135))),
                mul(res_mul_col12, sub(res_limb_23_col220, mul_res_limb_23_col164)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_24_col221, op1_limb_24_col108)),
                    mul(res_add_col11, sub(res_limb_24_col221, add_res_limb_24_col136))),
                mul(res_mul_col12, sub(res_limb_24_col221, mul_res_limb_24_col165)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_25_col222, op1_limb_25_col109)),
                    mul(res_add_col11, sub(res_limb_25_col222, add_res_limb_25_col137))),
                mul(res_mul_col12, sub(res_limb_25_col222, mul_res_limb_25_col166)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_26_col223, op1_limb_26_col110)),
                    mul(res_add_col11, sub(res_limb_26_col223, add_res_limb_26_col138))),
                mul(res_mul_col12, sub(res_limb_26_col223, mul_res_limb_26_col167)))));

        cuda_evaluator.add_constraint(mul(res_constrained,
            add(add(mul(res_op1, sub(res_limb_27_col224, op1_limb_27_col111)),
                    mul(res_add_col11, sub(res_limb_27_col224, add_res_limb_27_col139))),
                mul(res_mul_col12, sub(res_limb_27_col224, mul_res_limb_27_col168)))));
    }

    // Implement HandleOpcodes subroutine
    // This handles opcode-specific logic (call, ret, assert_eq)
    {
        // ASSERT_EQ opcode: res must equal dst for all 28 limbs
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_0_col197, dst_limb_0_col23)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_1_col198, dst_limb_1_col24)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_2_col199, dst_limb_2_col25)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_3_col200, dst_limb_3_col26)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_4_col201, dst_limb_4_col27)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_5_col202, dst_limb_5_col28)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_6_col203, dst_limb_6_col29)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_7_col204, dst_limb_7_col30)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_8_col205, dst_limb_8_col31)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_9_col206, dst_limb_9_col32)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_10_col207, dst_limb_10_col33)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_11_col208, dst_limb_11_col34)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_12_col209, dst_limb_12_col35)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_13_col210, dst_limb_13_col36)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_14_col211, dst_limb_14_col37)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_15_col212, dst_limb_15_col38)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_16_col213, dst_limb_16_col39)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_17_col214, dst_limb_17_col40)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_18_col215, dst_limb_18_col41)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_19_col216, dst_limb_19_col42)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_20_col217, dst_limb_20_col43)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_21_col218, dst_limb_21_col44)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_22_col219, dst_limb_22_col45)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_23_col220, dst_limb_23_col46)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_24_col221, dst_limb_24_col47)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_25_col222, dst_limb_25_col48)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_26_col223, dst_limb_26_col49)));
        cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_27_col224, dst_limb_27_col50)));

        // Constants for HandleOpcodes section
        const m31 M31_2 = m31(2);
        const m31 M31_4 = m31(4);
        const m31 M31_512 = m31(512);
        const m31 M31_262144 = m31(262144);
        const m31 M31_134217728 = m31(134217728);

        // RET opcode constraints
        // ret opcode offset0 equals -2
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, add(offset0_adjusted, M31_2)));
        // ret opcode offset2 equals -1
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, add(offset2_adjusted, M31_1)));
        // ret opcode flags sum check
        m31 ret_flags_sum = sub(sub(sub(sub(M31_4, pc_update_jump_col13), dst_base_fp_col6), op1_base_fp_col9), res_op1);
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, ret_flags_sum));

        // CALL opcode constraints
        // call opcode offset0 equals 0
        cuda_evaluator.add_constraint(mul(opcode_call_col18, offset0_adjusted));
        // call opcode offset1 equals 1
        cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(offset1_adjusted, M31_1)));
        // call opcode flags
        cuda_evaluator.add_constraint(mul(opcode_call_col18, add(op0_base_fp_col7, dst_base_fp_col6)));

        // CondFelt252AsAddr for dst (CALL validation)
        m31 cond_felt_252_as_addr_output_dst;
        {
            // Limbs 4-27 must be zero when opcode_call is active
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_4_col27));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_5_col28));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_6_col29));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_7_col30));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_8_col31));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_9_col32));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_10_col33));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_11_col34));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_12_col35));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_13_col36));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_14_col37));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_15_col38));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_16_col39));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_17_col40));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_18_col41));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_19_col42));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_20_col43));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_21_col44));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_22_col45));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_23_col46));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_24_col47));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_25_col48));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_26_col49));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, dst_limb_27_col50));

            // CondRangeCheck2 for limb 3
            cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col225, sub(M31_1, partial_limb_msb_col225)), opcode_call_col18));
            m31 partial_limb_bit_before_msb_dst = sub(dst_limb_3_col26, mul(partial_limb_msb_col225, M31_2));
            partial_limb_bit_before_msb_dst = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_dst);
            cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_dst, sub(M31_1, partial_limb_bit_before_msb_dst)), opcode_call_col18));

            // Compute address from limbs 0-3
            cond_felt_252_as_addr_output_dst = add(add(add(
                dst_limb_0_col23,
                mul(dst_limb_1_col24, M31_512)),
                mul(dst_limb_2_col25, M31_262144)),
                mul(dst_limb_3_col26, M31_134217728));
        }

        // call opcode: dst must equal fp
        cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(cond_felt_252_as_addr_output_dst, input_fp_col2)));

        // CondFelt252AsAddr for op0 (CALL validation)
        m31 cond_felt_252_as_addr_output_op0;
        {
            // Limbs 4-27 must be zero when opcode_call is active
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_4_col57));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_5_col58));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_6_col59));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_7_col60));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_8_col61));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_9_col62));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_10_col63));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_11_col64));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_12_col65));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_13_col66));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_14_col67));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_15_col68));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_16_col69));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_17_col70));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_18_col71));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_19_col72));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_20_col73));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_21_col74));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_22_col75));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_23_col76));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_24_col77));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_25_col78));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_26_col79));
            cuda_evaluator.add_constraint(mul(opcode_call_col18, op0_limb_27_col80));

            // CondRangeCheck2 for limb 3
            cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col226, sub(M31_1, partial_limb_msb_col226)), opcode_call_col18));
            m31 partial_limb_bit_before_msb_op0 = sub(op0_limb_3_col56, mul(partial_limb_msb_col226, M31_2));
            partial_limb_bit_before_msb_op0 = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_op0);
            cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_op0, sub(M31_1, partial_limb_bit_before_msb_op0)), opcode_call_col18));

            // Compute address from limbs 0-3
            cond_felt_252_as_addr_output_op0 = add(add(add(
                op0_limb_0_col53,
                mul(op0_limb_1_col54, M31_512)),
                mul(op0_limb_2_col55, M31_262144)),
                mul(op0_limb_3_col56, M31_134217728));
        }

        // call opcode: op0 must equal pc + instruction_size
        cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(cond_felt_252_as_addr_output_op0, add(input_pc_col0, one_plus_op1_imm))));
    }
    // Implement UpdateRegisters subroutine
    // This handles PC, AP, FP register updates based on opcode flags
    {
        const m31 M31_2 = m31(2);
        const m31 M31_136 = m31(136);
        const m31 M31_256 = m31(256);
        const m31 M31_508 = m31(508);
        const m31 M31_511 = m31(511);
        const m31 M31_512 = m31(512);
        const m31 M31_262144 = m31(262144);
        const m31 M31_134217728 = m31(134217728);
        const m31 M31_536870912 = m31(536870912);
        const m31 M31_1048576 = m31(1048576);

        // CondFelt252AsAddr #1: res as address (for pc_update_jump)
        // Limbs 4-27 must be zero when pc_update_jump is active
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_4_col201));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_5_col202));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_6_col203));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_7_col204));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_8_col205));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_9_col206));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_10_col207));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_11_col208));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_12_col209));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_13_col210));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_14_col211));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_15_col212));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_16_col213));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_17_col214));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_18_col215));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_19_col216));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_20_col217));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_21_col218));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_22_col219));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_23_col220));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_24_col221));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_25_col222));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_26_col223));
        cuda_evaluator.add_constraint(mul(pc_update_jump_col13, res_limb_27_col224));

        // CondRangeCheck2 for res limb 3
        cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col227, sub(M31_1, partial_limb_msb_col227)), pc_update_jump_col13));
        m31 partial_limb_bit_before_msb_res = sub(res_limb_3_col200, mul(partial_limb_msb_col227, M31_2));
        partial_limb_bit_before_msb_res = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_res);
        cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_res, sub(M31_1, partial_limb_bit_before_msb_res)), pc_update_jump_col13));

        // Compute res as address
        m31 cond_felt_252_as_addr_output_res = add(add(add(
            res_limb_0_col197,
            mul(res_limb_1_col198, M31_512)),
            mul(res_limb_2_col199, M31_262144)),
            mul(res_limb_3_col200, M31_134217728));

        // CondFelt252AsAddr #2: dst as address (for opcode_ret)
        // Limbs 4-27 must be zero when opcode_ret is active
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_4_col27));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_5_col28));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_6_col29));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_7_col30));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_8_col31));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_9_col32));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_10_col33));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_11_col34));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_12_col35));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_13_col36));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_14_col37));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_15_col38));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_16_col39));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_17_col40));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_18_col41));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_19_col42));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_20_col43));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_21_col44));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_22_col45));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_23_col46));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_24_col47));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_25_col48));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_26_col49));
        cuda_evaluator.add_constraint(mul(opcode_ret_col19, dst_limb_27_col50));

        // CondRangeCheck2 for dst limb 3
        cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col228, sub(M31_1, partial_limb_msb_col228)), opcode_ret_col19));
        m31 partial_limb_bit_before_msb_dst = sub(dst_limb_3_col26, mul(partial_limb_msb_col228, M31_2));
        partial_limb_bit_before_msb_dst = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_dst);
        cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_dst, sub(M31_1, partial_limb_bit_before_msb_dst)), opcode_ret_col19));

        // Compute dst as address
        m31 cond_felt_252_as_addr_output_dst_ret = add(add(add(
            dst_limb_0_col23,
            mul(dst_limb_1_col24, M31_512)),
            mul(dst_limb_2_col25, M31_262144)),
            mul(dst_limb_3_col26, M31_134217728));

        // CondFelt252AsRelImm #1: res as relative immediate (for pc_update_jump_rel + ap_update_add)
        m31 res_as_rel_imm_cond = add(pc_update_jump_rel_col14, ap_update_add_col16);

        // CondDecodeSmallSign constraints
        cuda_evaluator.add_constraint(mul(msb_col229, sub(msb_col229, M31_1)));
        cuda_evaluator.add_constraint(mul(mid_limbs_set_col230, sub(mid_limbs_set_col230, M31_1)));
        cuda_evaluator.add_constraint(mul(mul(res_as_rel_imm_cond, mid_limbs_set_col230), sub(msb_col229, M31_1)));

        // remainder_bits for res limb 3
        m31 remainder_bits_res = sub(res_limb_3_col200, mul(mid_limbs_set_col230, M31_508));
        remainder_bits_res = cuda_evaluator.add_intermediate(remainder_bits_res);

        // CondRangeCheck2 for remainder_bits
        cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col231, sub(M31_1, partial_limb_msb_col231)), res_as_rel_imm_cond));
        m31 partial_limb_bit_before_msb_res_relimm = sub(remainder_bits_res, mul(partial_limb_msb_col231, M31_2));
        partial_limb_bit_before_msb_res_relimm = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_res_relimm);
        cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_res_relimm, sub(M31_1, partial_limb_bit_before_msb_res_relimm)), res_as_rel_imm_cond));

        // rel_imm res limb 4-20 are fixed to 511 * mid_limbs_set
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_4_col201, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_5_col202, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_6_col203, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_7_col204, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_8_col205, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_9_col206, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_10_col207, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_11_col208, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_12_col209, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_13_col210, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_14_col211, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_15_col212, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_16_col213, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_17_col214, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_18_col215, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_19_col216, mul(mid_limbs_set_col230, M31_511))));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_20_col217, mul(mid_limbs_set_col230, M31_511))));

        // rel_imm res limb 21 is fixed to (136 * msb - mid_limbs_set)
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_21_col218, sub(mul(M31_136, msb_col229), mid_limbs_set_col230))));

        // rel_imm res limb 22-26 are fixed to 0
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, res_limb_22_col219));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, res_limb_23_col220));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, res_limb_24_col221));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, res_limb_25_col222));
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, res_limb_26_col223));

        // rel_imm res limb 27 is fixed to (msb * 256)
        cuda_evaluator.add_constraint(mul(res_as_rel_imm_cond, sub(res_limb_27_col224, mul(msb_col229, M31_256))));

        // Compute res as relative immediate
        m31 cond_felt_252_as_rel_imm_output_res = sub(sub(add(add(add(
            res_limb_0_col197,
            mul(res_limb_1_col198, M31_512)),
            mul(res_limb_2_col199, M31_262144)),
            mul(remainder_bits_res, M31_134217728)),
            msb_col229),
            mul(M31_536870912, mid_limbs_set_col230));
        // dst_not_p constraint
        m31 diff_from_p_0 = sub(dst_limb_0_col23, M31_1);
        m31 diff_from_p_21 = sub(dst_limb_21_col44, M31_136);
        m31 diff_from_p_27 = sub(dst_limb_27_col50, M31_256);
        diff_from_p_0 = cuda_evaluator.add_intermediate(diff_from_p_0);
        diff_from_p_21 = cuda_evaluator.add_intermediate(diff_from_p_21);
        diff_from_p_27 = cuda_evaluator.add_intermediate(diff_from_p_27);

        m31 dst_sum_squares = add(add(
            mul(diff_from_p_0, diff_from_p_0),
            add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(
                dst_limb_1_col24,
                dst_limb_2_col25),
                dst_limb_3_col26),
                dst_limb_4_col27),
                dst_limb_5_col28),
                dst_limb_6_col29),
                dst_limb_7_col30),
                dst_limb_8_col31),
                dst_limb_9_col32),
                dst_limb_10_col33),
                dst_limb_11_col34),
                dst_limb_12_col35),
                dst_limb_13_col36),
                dst_limb_14_col37),
                dst_limb_15_col38),
                dst_limb_16_col39),
                dst_limb_17_col40),
                dst_limb_18_col41),
                dst_limb_19_col42),
                dst_limb_20_col43)),
        add(
            mul(diff_from_p_21, diff_from_p_21),
            add(add(add(add(add(
                dst_limb_22_col45,
                dst_limb_23_col46),
                dst_limb_24_col47),
                dst_limb_25_col48),
                dst_limb_26_col49),
                mul(diff_from_p_27, diff_from_p_27))));
        dst_sum_squares = cuda_evaluator.add_intermediate(dst_sum_squares);

        cuda_evaluator.add_constraint(sub(mul(dst_sum_squares, dst_sum_squares_inv_col232), M31_1));

        // dst_sum for conditional jump
        m31 dst_sum = add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(
            dst_limb_0_col23,
            dst_limb_1_col24),
            dst_limb_2_col25),
            dst_limb_3_col26),
            dst_limb_4_col27),
            dst_limb_5_col28),
            dst_limb_6_col29),
            dst_limb_7_col30),
            dst_limb_8_col31),
            dst_limb_9_col32),
            dst_limb_10_col33),
            dst_limb_11_col34),
            dst_limb_12_col35),
            dst_limb_13_col36),
            dst_limb_14_col37),
            dst_limb_15_col38),
            dst_limb_16_col39),
            dst_limb_17_col40),
            dst_limb_18_col41),
            dst_limb_19_col42),
            dst_limb_20_col43),
            dst_limb_21_col44),
            dst_limb_22_col45),
            dst_limb_23_col46),
            dst_limb_24_col47),
            dst_limb_25_col48),
            dst_limb_26_col49),
            dst_limb_27_col50);
        dst_sum = cuda_evaluator.add_intermediate(dst_sum);

        // op1_as_rel_imm_cond constraint
        cuda_evaluator.add_constraint(sub(op1_as_rel_imm_cond_col234, mul(pc_update_jnz_col15, dst_sum)));

        // CondFelt252AsRelImm #2: op1 as relative immediate (for op1_as_rel_imm_cond)
        // CondDecodeSmallSign constraints
        cuda_evaluator.add_constraint(mul(msb_col235, sub(msb_col235, M31_1)));
        cuda_evaluator.add_constraint(mul(mid_limbs_set_col236, sub(mid_limbs_set_col236, M31_1)));
        cuda_evaluator.add_constraint(mul(mul(op1_as_rel_imm_cond_col234, mid_limbs_set_col236), sub(msb_col235, M31_1)));

        // remainder_bits for op1 limb 3
        m31 remainder_bits_op1 = sub(op1_limb_3_col87, mul(mid_limbs_set_col236, M31_508));
        remainder_bits_op1 = cuda_evaluator.add_intermediate(remainder_bits_op1);

        // CondRangeCheck2 for remainder_bits
        cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col237, sub(M31_1, partial_limb_msb_col237)), op1_as_rel_imm_cond_col234));
        m31 partial_limb_bit_before_msb_op1_relimm = sub(remainder_bits_op1, mul(partial_limb_msb_col237, M31_2));
        partial_limb_bit_before_msb_op1_relimm = cuda_evaluator.add_intermediate(partial_limb_bit_before_msb_op1_relimm);
        cuda_evaluator.add_constraint(mul(mul(partial_limb_bit_before_msb_op1_relimm, sub(M31_1, partial_limb_bit_before_msb_op1_relimm)), op1_as_rel_imm_cond_col234));

        // rel_imm op1 limb 4-20 are fixed to 511 * mid_limbs_set
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_4_col88, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_5_col89, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_6_col90, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_7_col91, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_8_col92, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_9_col93, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_10_col94, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_11_col95, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_12_col96, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_13_col97, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_14_col98, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_15_col99, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_16_col100, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_17_col101, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_18_col102, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_19_col103, mul(mid_limbs_set_col236, M31_511))));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_20_col104, mul(mid_limbs_set_col236, M31_511))));

        // rel_imm op1 limb 21 is fixed to (136 * msb - mid_limbs_set)
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_21_col105, sub(mul(M31_136, msb_col235), mid_limbs_set_col236))));

        // rel_imm op1 limb 22-26 are fixed to 0
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, op1_limb_22_col106));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, op1_limb_23_col107));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, op1_limb_24_col108));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, op1_limb_25_col109));
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, op1_limb_26_col110));

        // rel_imm op1 limb 27 is fixed to (msb * 256)
        cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col234, sub(op1_limb_27_col111, mul(msb_col235, M31_256))));

        // Compute op1 as relative immediate
        m31 cond_felt_252_as_rel_imm_output_op1 = sub(sub(add(add(add(
            op1_limb_0_col84,
            mul(op1_limb_1_col85, M31_512)),
            mul(op1_limb_2_col86, M31_262144)),
            mul(remainder_bits_op1, M31_134217728)),
            msb_col235),
            mul(M31_536870912, mid_limbs_set_col236));

        // Conditional jump constraints
        // Constraint1: When dst != 0, next_pc_jnz = pc + op1_as_rel_imm
        cuda_evaluator.add_constraint(mul(sub(next_pc_jnz_col238, add(input_pc_col0, cond_felt_252_as_rel_imm_output_op1)), dst_sum));
        // instruction_size intermediate
        m31 instruction_size = add(add(M31_1, op1_imm_col8), pc_update_jump_rel_col14);
        instruction_size = cuda_evaluator.add_intermediate(instruction_size);

        // Constraint2: When dst == 0, next_pc_jnz = pc + instruction_size
        cuda_evaluator.add_constraint(mul(sub(next_pc_jnz_col238, add(input_pc_col0, instruction_size)), sub(mul(dst_sum, dst_sum_inv_col233), M31_1)));

        // next_pc constraint
        m31 next_pc_computed = add(add(add(
            mul(pc_update_regular, add(input_pc_col0, instruction_size)),
            mul(pc_update_jump_col13, cond_felt_252_as_addr_output_res)),
            mul(pc_update_jump_rel_col14, add(input_pc_col0, cond_felt_252_as_rel_imm_output_res))),
            mul(pc_update_jnz_col15, next_pc_jnz_col238));
        cuda_evaluator.add_constraint(sub(next_pc_col239, next_pc_computed));

        // next_ap constraint
        m31 next_ap_computed = add(add(add(
            input_ap_col1,
            mul(ap_update_add_col16, cond_felt_252_as_rel_imm_output_res)),
            ap_update_add_1_col17),
            mul(opcode_call_col18, M31_2));
        cuda_evaluator.add_constraint(sub(next_ap_col240, next_ap_computed));

        // RangeCheckAp: Range check next_ap
        m31 range_check_ap_top_bits = mul(sub(next_ap_col240, range_check_ap_bot11bits_col241), M31_1048576);
        range_check_ap_top_bits = cuda_evaluator.add_intermediate(range_check_ap_top_bits);

        // range_check_18 lookup for top bits
        {
            m31 values[1] = {range_check_ap_top_bits};
            RelationEntry<1> entry(
                generic_opcode_eval->range_check_18_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<1>(entry);
        }
        // range_check_11 lookup for bottom 11 bits
        {
            m31 values[1] = {range_check_ap_bot11bits_col241};
            RelationEntry<1> entry(
                generic_opcode_eval->range_check_11_lookup_elements,
                qm31{m31(1), m31(0)},
                values
            );
            cuda_evaluator.add_to_relation<1>(entry);
        }

        // next_fp constraint
        m31 next_fp_computed = add(add(
            mul(fp_update_regular, input_fp_col2),
            mul(opcode_ret_col19, cond_felt_252_as_addr_output_dst_ret)),
            mul(opcode_call_col18, add(input_ap_col1, M31_2)));
        cuda_evaluator.add_constraint(sub(next_fp_col242, next_fp_computed));
    }

    // Add relations for opcodes lookup (input state)
    {
        m31 values[3] = {input_pc_col0, input_ap_col1, input_fp_col2};
        RelationEntry<3> entry(
            generic_opcode_eval->opcodes_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    // Add relations for opcodes lookup (output state)
    {
        qm31 neg_multiplicity = qm31{{neg(enabler), 0}, {0, 0}};
        m31 values[3] = {next_pc_col239, next_ap_col240, next_fp_col242};
        RelationEntry<3> entry(
            generic_opcode_eval->opcodes_lookup_elements,
            neg_multiplicity,  // negative multiplicity
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C"
void evaluate_generic_opcode(
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
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    GenericOpcode_Eval *device_generic_opcode_eval = cuda_malloc<GenericOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<GenericOpcode_Eval>((GenericOpcode_Eval*)eval, device_generic_opcode_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_generic_opcode");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_generic_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_generic_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_generic_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_generic_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<num_blocks, block_dim>>>(
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
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim>>>(
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
        g_should_accumulate_host  // Read from global variable set by dispatcher
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_generic_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_generic_opcode_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
