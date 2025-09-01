#include "fields.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include "timer.cuh"

__global__ void evaluate_wide_fibonacci_constraint_quotients_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *numerators,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns
) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned eval_domain_size = 1 << (eval_domain_log_size);

    if (row_index < eval_domain_size) {
        unsigned col_index[3] = {0, 0, 0};
        unsigned constraint_index = 0;
        qm31 row_res = {0};

        m31 a = next_trace_mask(trace1_evaluations, col_index, row_index, domain_log_size, eval_domain_log_size);
        m31 b = next_trace_mask(trace1_evaluations, col_index, row_index, domain_log_size, eval_domain_log_size);

        for (unsigned instance_index = 2; instance_index < number_of_columns + 2; instance_index++) {
            m31 c = next_trace_mask(trace1_evaluations, col_index, row_index, domain_log_size, eval_domain_log_size);

            add_constraint(random_coeff_powers, &constraint_index, sub(c, add(square(a), square(b))), &row_res);
            a = b;
            b = c;
        }
        numerators[row_index] = row_res;
        m31 denom_inv = denominator_inverses[row_index >> domain_log_size];
        qm31 constraint_quotient = mul(
            denom_inv,
            numerators[row_index]
        );

        quotients_0[row_index] = constraint_quotient.a.a;
        quotients_1[row_index] = constraint_quotient.a.b;
        quotients_2[row_index] = constraint_quotient.b.a;
        quotients_3[row_index] = constraint_quotient.b.b;

    }
}

void evaluate_wide_fibonacci_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns
) {
    unsigned eval_domain_size = 1 << (eval_domain_log_size);
    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    qm31 *numerators = (qm31*) cuda_alloc_zeroes_uint32_t(4 * eval_domain_size);

    int block_dim = eval_domain_size < THREAD_COUNT_MAX ? eval_domain_size : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (eval_domain_size + block_dim - 1) / block_dim;

    timer global_timer;
    // global_timer.start("evaluate_wide_fibonacci_constraint_quotients_on_domain");
    evaluate_wide_fibonacci_constraint_quotients_kernel<<<num_blocks, block_dim>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        device_trace0_evaluations,
        device_trace1_evaluations,
        numerators,
        random_coeff_powers,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        number_of_columns
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // global_timer.end("evaluate_wide_fibonacci_constraint_quotients_on_domain");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(numerators);
}

__global__ void generate_wide_fibonacci_trace_kernel(
    m31 *input_a,
    m31 *input_b,
    unsigned input_len,
    m31 **device_trace,
    unsigned columns
) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < input_len) {
        device_trace[0][row_index] = input_a[row_index];
        device_trace[1][row_index] = input_b[row_index];
        for (int i = 2; i < columns; i++) {
            device_trace[i][row_index] = add(square(device_trace[i - 2][row_index]), square(device_trace[i - 1][row_index]));
        }
    }
}

void generate_wide_fibonacci_trace(
    m31 *input_a,
    m31 *input_b,
    unsigned input_len,
    m31 **traces,
    unsigned traces_len,
    unsigned n_columns
) {
    m31 **device_trace = clone_to_device<m31*>(traces, traces_len);

    int block_dim = input_len < THREAD_COUNT_MAX ? input_len : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_len + block_dim - 1) / block_dim;

    generate_wide_fibonacci_trace_kernel<<<num_blocks, block_dim>>>(
        input_a,
        input_b,
        input_len,
        device_trace,
        n_columns
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace);
}