#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_assert_eq_opcode_double_deref.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_mem_verify_equal.cuh"
#include "evaluate_read_positive_num_bits.cuh"

#define ADD_CODE_SMALL_THREAD_COUNT_MAX 256

__launch_bounds__(256, 2)
__global__ void evaluate_assert_eq_double_deref_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    AssertEqDoubleDerefEval *assert_eq_double_deref_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    CudaAssertEvaluator cuda_evaluator(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {{0,0},{0,0}},
        0,
        {{0,0},{0,0}},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );
    m31 input_pc_col0            = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1            = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2            = cuda_evaluator.next_trace_mask();
    m31 offset0_col3             = cuda_evaluator.next_trace_mask();
    m31 offset1_col4             = cuda_evaluator.next_trace_mask();
    m31 offset2_col5             = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col6         = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col7         = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col8     = cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col9        = cuda_evaluator.next_trace_mask();
    m31 mem0_base_col10          = cuda_evaluator.next_trace_mask();
    m31 mem1_base_id_col11       = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_0_col12   = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_1_col13   = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_2_col14   = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_3_col15   = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col16   = cuda_evaluator.next_trace_mask();
    m31 dst_id_col17             = cuda_evaluator.next_trace_mask();
    m31 enabler                  = cuda_evaluator.next_trace_mask();

    const m31 M31_1 = m31(1);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);
    const m31 M31_134217728 = m31(134217728);

    // enabler is boolean
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // --- DecodeInstructionCb32B ---
    m31 decode_instruction_cb32b_output_tmp_b1151_8[19] = {0};
    evaluate_decode_instruction_cb32b(
        input_pc_col0,
        offset0_col3,
        offset1_col4,
        offset2_col5,
        dst_base_fp_col6,
        op0_base_fp_col7,
        ap_update_add_1_col8,
        decode_instruction_cb32b_output_tmp_b1151_8,
        assert_eq_double_deref_eval->verify_instruction_lookup_elements,
        &cuda_evaluator
    );

    // mem_dst_base
    cuda_evaluator.add_constraint(
        sub(
            mem_dst_base_col9,
            add(
                mul(dst_base_fp_col6, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
            )
        )
    );
    // mem0_base
    cuda_evaluator.add_constraint(
        sub(
            mem0_base_col10,
            add(
                mul(op0_base_fp_col7, input_fp_col2),
                mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
            )
        )
    );

    // --- ReadPositiveNumBits29 ---
    m31 read_positive_num_bits_29_output_tmp_b1151_11[29] = {0};
    evaluate_read_positive_num_bits_29(
        add(mem0_base_col10, decode_instruction_cb32b_output_tmp_b1151_8[1]),
        mem1_base_id_col11,
        mem1_base_limb_0_col12,
        mem1_base_limb_1_col13,
        mem1_base_limb_2_col14,
        mem1_base_limb_3_col15,
        partial_limb_msb_col16,
        read_positive_num_bits_29_output_tmp_b1151_11,
        assert_eq_double_deref_eval->memory_address_to_id_lookup_elements,
        assert_eq_double_deref_eval->memory_id_to_big_lookup_elements,
        &cuda_evaluator
    );

    // --- MemVerifyEqual ---
    m31 mem_verify_equal_inputs[2] = {
        add(mem_dst_base_col9, decode_instruction_cb32b_output_tmp_b1151_8[0]),
        add(
            add(
                add(
                    mem1_base_limb_0_col12,
                    mul(mem1_base_limb_1_col13, M31_512)
                ),
                mul(mem1_base_limb_2_col14, M31_262144)
            ),
            add(
                mul(mem1_base_limb_3_col15, M31_134217728),
                decode_instruction_cb32b_output_tmp_b1151_8[2]
            )
        )
    };
    evaluate_mem_verify_equal(
        mem_verify_equal_inputs[0],
        mem_verify_equal_inputs[1],
        dst_id_col17,
        assert_eq_double_deref_eval->memory_address_to_id_lookup_elements,
        &cuda_evaluator
    );

    // --- Lookup: opcodes_0 ---
    {
        m31 values[3] = {input_pc_col0, input_ap_col1, input_fp_col2};
        RelationEntry<3> entry(
            assert_eq_double_deref_eval->opcode_lookup_elements,
            qm31{enabler},
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }
    // --- Lookup: opcodes_1 ---
    {
        m31 values[3] = {
            add(input_pc_col0, M31_1),
            add(input_ap_col1, ap_update_add_1_col8),
            input_fp_col2
        };
        RelationEntry<3> entry(
            assert_eq_double_deref_eval->opcode_lookup_elements,
            (enabler == 0 ? qm31{0} : qm31{(P - enabler), 0 , 0 , 0}),
            values
        );
        cuda_evaluator.add_to_relation<3>(entry);
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;

}

__launch_bounds__(256, 2)
__global__ void assert_eq_double_deref_process_constraint_post_kernel(
    qm31 *numerators,
    Fraction *intermediate_fractions,
    unsigned *constraint_index_array,
    m31 **trace2_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int logup_counts,
    unsigned int last_batch,
    qm31 cumsum_shift
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    const unsigned logup_interaction = 2;

    // CudaEvaluator evaluator(
    //     trace2_evaluations,
    //     random_coeff_powers,
    //     domain_log_size,
    //     eval_domain_log_size,
    //     numerators[row],
    //     constraint_index_array[row],
    //     row
    // );
    CudaAssertEvaluator evaluator(
        trace2_evaluations,
        random_coeff_powers,
        constraint_index_array[row],
        row,
        {{0,0},{0,0}},
        logup_interaction,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    qm31 prev_col_cumsum = { {0, 0}, {0, 0} };

    for (unsigned i = 0; i < last_batch; ++i) {
        // if (row == 1 && i == 0) {
        //     for (unsigned j = 0; j < logup_counts; j++) {
        //         printf("row:%d, j:%d ,intermediate_fractions[%d]:{ numerator: (%d + %di) + (%d + %di)u, denominator: (%d + %di) + (%d + %di)u }\n", row, j, j, intermediate_fractions[j + row * logup_counts].numerator.a.a, intermediate_fractions[j + row * logup_counts].numerator.a.b, intermediate_fractions[j + row * logup_counts].numerator.b.a, intermediate_fractions[j + row * logup_counts].numerator.b.b, intermediate_fractions[j + row * logup_counts].denominator.a.a, intermediate_fractions[j + row * logup_counts].denominator.a.b, intermediate_fractions[j + row * logup_counts].denominator.b.a, intermediate_fractions[j + row * logup_counts].denominator.b.b);
        //     }
        // }
        const Fraction cur_frac = Fraction::sum(&intermediate_fractions[2 * i + row * logup_counts], 2);
        // if (row == 1) {
        //     printf("row:%d, i:%d ,cur_frac.numerator: (%d, %d, %d, %d), cur_frac.denominator: (%d, %d, %d, %d)\n", row, i, cur_frac.numerator.a.a, cur_frac.numerator.a.b, cur_frac.numerator.b.a, cur_frac.numerator.b.b, cur_frac.denominator.a.a, cur_frac.denominator.a.b, cur_frac.denominator.b.a, cur_frac.denominator.b.b);
        // }
        qm31 cur_cumsum_arr[1] = { { {0, 0}, {0, 0} } };
        int offsets[1] = { 0 };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets, 2, cur_cumsum_arr);
        const qm31 cur_cumsum = cur_cumsum_arr[0];
        const qm31 diff = sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;
        const qm31 constrain_val = sub(mul(diff, cur_frac.denominator), cur_frac.numerator);
        evaluator.add_constraint_ext(constrain_val);
    }
    {
        const Fraction frac_sum = Fraction::sum(&intermediate_fractions[last_batch * 2 + row * logup_counts], 1);
        // if (row == 1) {
        //     printf("row:%d, last_batch:%d, frac_sum: numerator: { numerator: (%d + %di) + (%d + %di)u, denominator: (%d + %di) + (%d + %di)u \n", row, last_batch, frac_sum.numerator.a.a, frac_sum.numerator.a.b, frac_sum.numerator.b.a, frac_sum.numerator.b.b, frac_sum.denominator.a.a, frac_sum.denominator.a.b, frac_sum.denominator.b.a, frac_sum.denominator.b.b);
        // }
        int offsets2[2] = { 0, -1 };
        qm31 cumsum2[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets2, 2, cumsum2);
        const qm31 prev_row_cumsum = cumsum2[1];
        const qm31 cur_cumsum = cumsum2[0];
        // if (row == 1) {
        //     printf("row:%d, last_batch:%d, prev_row_cumsum: (%d + %di) + (%d + %di)u, cur_cumsum: (%d + %di) + (%d + %di)u \n", row, last_batch, prev_row_cumsum.a.a, prev_row_cumsum.a.b, prev_row_cumsum.b.a, prev_row_cumsum.b.b, cur_cumsum.a.a, cur_cumsum.a.b, cur_cumsum.b.a, cur_cumsum.b.b);
        // }
        const qm31 diff = sub(sub(cur_cumsum, prev_row_cumsum), prev_col_cumsum);
        const qm31 fixed_diff = add(diff, cumsum_shift);
        const qm31 constrain_val = sub(mul(fixed_diff, frac_sum.denominator), frac_sum.numerator);
        evaluator.add_constraint_ext(constrain_val);
    }
    // numerators[row] = evaluator.row_res;
}

__global__ void assert_eq_double_deref_evaluate_constraint_quotients_finalize_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    qm31 *numerators,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    const m31 denom_inv = denominator_inverses[row >> domain_log_size];
    const qm31 row_numer = numerators[row];
    const qm31 constraint_quotient = mul(denom_inv, row_numer);

    quotients_0[row] = constraint_quotient.a.a;
    quotients_1[row] = constraint_quotient.a.b;
    quotients_2[row] = constraint_quotient.b.a;
    quotients_3[row] = constraint_quotient.b.b;
}

void evaluate_assert_eq_opcode_double_deref(
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
    AssertEqDoubleDerefEval *assert_eq_eval = (AssertEqDoubleDerefEval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    AssertEqDoubleDerefEval *device_assert_eq_eval = cuda_malloc<AssertEqDoubleDerefEval>(1);
    cuda_mem_copy_host_to_device<AssertEqDoubleDerefEval>(assert_eq_eval, device_assert_eq_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constrain_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_assert_eq");

    int block_dim = eval_domain_size < ADD_CODE_SMALL_THREAD_COUNT_MAX ? eval_domain_size : ADD_CODE_SMALL_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;
    evaluate_assert_eq_double_deref_pre_kernel<<<num_blocks, block_dim>>>(
        numerators,
        device_trace1_evaluations,
        random_coeff_powers,
        domain_log_size,
        eval_domain_log_size,
        number_of_columns,
        device_assert_eq_eval,
        cumsum_shift,
        d_intermediate_fractions,
        logup_counts,
        constrain_index_array
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];
    printf("last batch: %d\n", last_batch);
    assert_eq_double_deref_process_constraint_post_kernel<<<num_blocks, block_dim>>>(
        numerators,
        d_intermediate_fractions,
        constrain_index_array,
        device_trace2_evaluations,
        random_coeff_powers,
        domain_log_size,
        eval_domain_log_size,
        logup_counts,
        last_batch,
        cumsum_shift
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    assert_eq_double_deref_evaluate_constraint_quotients_finalize_kernel<<<num_blocks, block_dim>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_assert_eq");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_assert_eq_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constrain_index_array);
}