#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_jump_opcode_rel.cuh"
#include "evaluate_read_small.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define JUMP_OPCODE_REL_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_jump_opcode_rel_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    JumpOpcodeRel_Eval *jump_opcode_rel_eval,
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

    // Load all 17 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset2_col3 = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col4 = cuda_evaluator.next_trace_mask();
    m31 op1_base_ap_col5 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col6 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_col7 = cuda_evaluator.next_trace_mask();
    m31 next_pc_id_col8 = cuda_evaluator.next_trace_mask();
    m31 msb_col9 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col10 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_0_col11 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_1_col12 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_2_col13 = cuda_evaluator.next_trace_mask();
    m31 remainder_bits_col14 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col15 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);

    // Constraint: enabler^2 = enabler (boolean constraint)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // DecodeInstruction3B105
    m31 decode_output[19];
    evaluate_decode_instruction_3b105(
        input_pc_col0,
        offset2_col3,
        op1_base_fp_col4,
        op1_base_ap_col5,
        ap_update_add_1_col6,
        decode_output,
        jump_opcode_rel_eval->verify_instruction_lookup_elements,
        &cuda_evaluator
    );

    // Constraint: Either flag op1_base_fp is on or flag op1_base_ap is on
    cuda_evaluator.add_constraint(sub(add(op1_base_fp_col4, op1_base_ap_col5), M31_1));

    // Constraint: mem1_base = op1_base_fp * input_fp + op1_base_ap * input_ap
    m31 mem1_base_expected = add(
        mul(op1_base_fp_col4, input_fp_col2),
        mul(op1_base_ap_col5, input_ap_col1)
    );
    cuda_evaluator.add_constraint(sub(mem1_base_col7, mem1_base_expected));

    // Read relative offset from [mem1_base + offset2] (ReadSmall)
    // decode_output[2] contains offset2 (offset2_col - 32768) from DecodeInstruction3B105
    m31 read_small_output[2] = {0};
    evaluate_read_small(
        add(mem1_base_col7, decode_output[2]),  // Address: mem1_base + offset2
        next_pc_id_col8,
        msb_col9,
        mid_limbs_set_col10,
        next_pc_limb_0_col11,
        next_pc_limb_1_col12,
        next_pc_limb_2_col13,
        remainder_bits_col14,
        partial_limb_msb_col15,
        read_small_output,
        jump_opcode_rel_eval->memory_address_to_id_lookup_elements,
        jump_opcode_rel_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator
    );

    // read_small_output[0] contains the relative offset
    // next_pc = input_pc + relative_offset

    // Add first opcodes relation entry (positive)
    {
        m31 values[3] = {
            input_pc_col0,
            input_ap_col1,
            input_fp_col2
        };
        RelationEntry<3> entry(
            jump_opcode_rel_eval->opcode_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    // Add second opcodes relation entry (negative)
    // next_pc = input_pc + relative_offset
    // next_ap = input_ap + ap_update_add_1
    // next_fp = input_fp (unchanged for jump)
    {
        m31 values[3] = {
            add(input_pc_col0, read_small_output[0]),  // next_pc = input_pc + offset
            add(input_ap_col1, ap_update_add_1_col6),
            input_fp_col2
        };
        // Negate the multiplicity for the second entry
        RelationEntry<3> entry(
            jump_opcode_rel_eval->opcode_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_jump_opcode_rel(
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

    JumpOpcodeRel_Eval *device_jump_rel_eval = cuda_malloc<JumpOpcodeRel_Eval>(1);
    cuda_mem_copy_host_to_device<JumpOpcodeRel_Eval>((JumpOpcodeRel_Eval*)eval, device_jump_rel_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_jump_opcode_rel");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_jump_opcode_rel_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_rel_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_jump_opcode_rel_pre_kernel<CudaEvaluator><<<num_blocks, block_dim>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_rel_eval,
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
    global_timer.end("evaluate_jump_opcode_rel");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_jump_rel_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
