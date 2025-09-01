#include "evaluate_constraints.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include "timer.cuh"
#include "utils.cuh"
#include "fields.cuh"

__global__ void evaluate_constraint_quotients_on_domain_new_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *constraints_vec,
    m31 *random_coeff_powers_vec,
    unsigned int constraints_coeff_pair_col_num,
    unsigned int constraints_coeff_pair_row_num,
    unsigned int trace_domain_log_size
) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < constraints_coeff_pair_row_num) {
        m31 denom_inv = denominator_inverses[row_index >> trace_domain_log_size];

        qm31 row_res = qm31{0};
        qm31* row_constraints = (qm31*)(constraints_vec + row_index * constraints_coeff_pair_col_num * 4);
        qm31* row_random_coeffs = (qm31*)(random_coeff_powers_vec + row_index * constraints_coeff_pair_col_num * 4);

        for (int col_index = 0; col_index < constraints_coeff_pair_col_num; col_index++) {
            qm31 constraint = row_constraints[col_index];
            qm31 random_coeff = row_random_coeffs[constraints_coeff_pair_col_num - col_index - 1];
            // if (row_index == 0) {
            //     printf("col_index: %d", col_index);
            //     print_qm31_device(constraint, "constraint:");
            //     print_qm31_device(random_coeff, "random_coeff:");
            // }
            qm31 mul_res = mul(constraint, random_coeff);
            row_res = add(row_res, mul_res);
        }
        // if (row_index == 0) {
        //     print_qm31_device(row_res, "row_res:");
        // }


        row_res = mul(row_res, qm31{{denom_inv, 0}, {0, 0}});

        qm31 current_quotient = {
            quotients_0[row_index],
            quotients_1[row_index],
            quotients_2[row_index],
            quotients_3[row_index]
        };

        current_quotient = add(current_quotient, row_res);

        quotients_0[row_index] = current_quotient.a.a;
        quotients_1[row_index] = current_quotient.a.b;
        quotients_2[row_index] = current_quotient.b.a;
        quotients_3[row_index] = current_quotient.b.b;
    }
}

void evaluate_constraint_quotients_on_domain_new(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *constraints_vec,
    m31 *random_coeff_powers_vec,
    unsigned int constraints_coeff_pair_col_num,
    unsigned int constraints_coeff_pair_row_num,
    unsigned int trace_domain_log_size
) {
    int block_dim = constraints_coeff_pair_row_num < THREAD_COUNT_MAX ? constraints_coeff_pair_row_num : THREAD_COUNT_MAX;
    int num_blocks = (constraints_coeff_pair_row_num + block_dim - 1) / block_dim;

    timer global_timer;
    global_timer.start("evaluate_constraint_quotients_on_domain_new");
    evaluate_constraint_quotients_on_domain_new_kernel<<<num_blocks, block_dim>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        denominator_inverses,
        constraints_vec,
        random_coeff_powers_vec,
        constraints_coeff_pair_col_num,
        constraints_coeff_pair_row_num,
        trace_domain_log_size
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_constraint_quotients_on_domain_new");
}

/// Ultra-optimized SIMD kernel with shared memory and reduced global memory access
/// Uses shared memory for coefficients and optimized memory access patterns
__global__ void evaluate_constraint_quotients_on_domain_new_simd_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *simd_constraints_vec,
    m31 *simd_random_coeff_powers_vec,
    unsigned int constraints_num,
    unsigned int vec_rows_num,
    unsigned int trace_domain_log_size
) {
    // Thread mapping: blockIdx.x = vec_row, threadIdx.x = lane (0-31)
    int vec_row_idx = blockIdx.x;
    int lane_idx = threadIdx.x;

    if (vec_row_idx >= vec_rows_num || lane_idx >= 32) {
        return;
    }

    // Calculate actual row index in the flattened domain
    int actual_row_idx = vec_row_idx * 32 + lane_idx;

    // Get denominator inverse for this row - cache locally
    m31 denom_inv = denominator_inverses[actual_row_idx >> trace_domain_log_size];

    // Accumulator for this row's constraint evaluation
    qm31 row_res = qm31{0};

    // Shared memory for random coefficients to reduce global memory access
    __shared__ qm31 shared_random_coeffs[1024]; // Support up to 1024 constraints

    // Load random coefficients into shared memory (coalesced access)
    int coeffs_per_thread = (constraints_num + 31) / 32;
    for (int i = 0; i < coeffs_per_thread; i++) {
        int coeff_idx = i * 32 + lane_idx;
        if (coeff_idx < constraints_num) {
            // Load coefficient in reverse order
            int actual_coeff_idx = constraints_num - coeff_idx - 1;
            int coeff_base_offset = actual_coeff_idx * 4;
            shared_random_coeffs[coeff_idx] = {
                {simd_random_coeff_powers_vec[coeff_base_offset + 0],     // a.a
                 simd_random_coeff_powers_vec[coeff_base_offset + 1]},    // a.b
                {simd_random_coeff_powers_vec[coeff_base_offset + 2],     // b.a
                 simd_random_coeff_powers_vec[coeff_base_offset + 3]}     // b.b
            };
        }
    }

    // Synchronize to ensure all coefficients are loaded
    __syncthreads();

    // Pre-calculate base constraint offset for this vec_row
    int vec_row_base_offset = vec_row_idx * constraints_num * 4 * 32;

    // Process constraints in blocks to improve memory access patterns
    const int CONSTRAINT_BLOCK_SIZE = 8;
    for (int block_start = 0; block_start < constraints_num; block_start += CONSTRAINT_BLOCK_SIZE) {
        int block_end = min(block_start + CONSTRAINT_BLOCK_SIZE, constraints_num);

        // Process constraints in this block
        for (int constraint_idx = block_start; constraint_idx < block_end; constraint_idx++) {
            // Calculate memory offset for this constraint in SIMD data
            // Optimized: Use pre-calculated base offset
            int constraint_base_offset = vec_row_base_offset + constraint_idx * 4 * 32;

            // Extract the 4 components of SecureField for this lane
            // Optimized: Reduce pointer arithmetic
            m31* constraint_ptr = simd_constraints_vec + constraint_base_offset + lane_idx;
            qm31 constraint = {
                {constraint_ptr[0],      // a.a
                 constraint_ptr[32]},    // a.b
                {constraint_ptr[64],     // b.a
                 constraint_ptr[96]}     // b.b
            };

            // Get random coefficient from shared memory
            qm31 random_coeff = shared_random_coeffs[constraint_idx];

            // Multiply constraint by random coefficient and accumulate
            qm31 mul_res = mul(constraint, random_coeff);
            row_res = add(row_res, mul_res);
        }
    }

    // Apply denominator inverse
    row_res = mul(row_res, qm31{{denom_inv, 0}, {0, 0}});

    // Read current quotient value and add our result
    // Optimized: Use single memory access pattern
    qm31 current_quotient = {
        {quotients_0[actual_row_idx], quotients_1[actual_row_idx]},
        {quotients_2[actual_row_idx], quotients_3[actual_row_idx]}
    };

    current_quotient = add(current_quotient, row_res);

    // Write back to global memory with coalesced access
    quotients_0[actual_row_idx] = current_quotient.a.a;
    quotients_1[actual_row_idx] = current_quotient.a.b;
    quotients_2[actual_row_idx] = current_quotient.b.a;
    quotients_3[actual_row_idx] = current_quotient.b.b;
}

void evaluate_constraint_quotients_on_domain_new_simd(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *simd_constraints_vec,
    m31 *simd_random_coeff_powers_vec,
    unsigned int constraints_num,
    unsigned int vec_rows_num,
    unsigned int trace_domain_log_size
) {
    // Optimized launch configuration with better occupancy
    int block_dim = 32;  // Fixed block size for SIMD lanes
    int num_blocks = vec_rows_num;

    // Calculate shared memory requirements
    size_t shared_mem_size = min(constraints_num, 1024) * sizeof(qm31);

    printf("Ultra-optimized SIMD kernel: %d blocks × %d threads = %d total threads\n",
           num_blocks, block_dim, num_blocks * block_dim);
    printf("Processing %d constraints per thread, shared memory: %zu bytes\n",
           constraints_num, shared_mem_size);

    timer global_timer;
    global_timer.start("evaluate_constraint_quotients_on_domain_new_simd");

    // Launch kernel with optimized configuration
    printf("Launching ultra-optimized SIMD CUDA kernel...\n");

    // Use CUDA streams for better performance if available
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    evaluate_constraint_quotients_on_domain_new_simd_kernel<<<num_blocks, block_dim, shared_mem_size, stream>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        denominator_inverses,
        simd_constraints_vec,
        simd_random_coeff_powers_vec,
        constraints_num,
        vec_rows_num,
        trace_domain_log_size
    );

    // Synchronize with optimized error handling
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cudaStreamDestroy(stream);
    printf("Ultra-optimized GPU kernel execution completed\n");

    global_timer.end("evaluate_constraint_quotients_on_domain_new_simd");
}

void evaluate_constraint_quotients_on_domain(
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
    void *eval,
    qm31 cumsum_shift
) {

    CommonEval *common_eval = (CommonEval *) eval;

    if (common_eval->eval_id == 1) {
        printf("call eval wide fib kernel\n");
        evaluate_wide_fibonacci_constraint_quotients_on_domain(
            quotients_0, quotients_1, quotients_2, quotients_3,
            trace0_evaluations,
            trace0_evaluations_len,
            trace1_evaluations,
            trace1_evaluations_len,
            random_coeff_powers,
            denominator_inverses,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns
        );
    } else if (common_eval->eval_id == 2) {
        printf("call eval poseidon kernel\n");
        evaluate_poseidon_constraint_quotients_on_domain(
            quotients_0, quotients_1, quotients_2, quotients_3,
            trace0_evaluations,
            trace0_evaluations_len,
            trace1_evaluations,
            trace1_evaluations_len,
            trace2_evaluations,
            trace2_evaluations_len,
            random_coeff_powers,
            denominator_inverses,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            eval,
            cumsum_shift
        );
    } else {
        printf("eval id:%d not supported\n", common_eval->eval_id);
    }

}