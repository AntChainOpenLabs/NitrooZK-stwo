#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include <stdint.h>

#include "gen_verify_bitwise_xor_trace.cuh"


__global__ void verify_bitwise_xor_4_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_4_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_4_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_4_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}

__global__ void verify_bitwise_xor_7_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_7_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_7_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_7_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}

__global__ void verify_bitwise_xor_8_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_8_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_8_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_8_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}

__global__ void verify_bitwise_xor_9_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_9_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_9_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_9_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}


__global__ void verify_bitwise_xor_12_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        m31 a = m31 {inputs[0][row]};
        m31 b = m31 {inputs[1][row]};

        uint32_t al = (uint32_t)(a & ((1ULL << VERIFY_BITWISE_12_LIMB_BITS) - 1));
        uint32_t ah = (uint32_t)(a >> VERIFY_BITWISE_12_LIMB_BITS);
        uint32_t bl = (uint32_t)(b & ((1ULL << VERIFY_BITWISE_12_LIMB_BITS) - 1));
        uint32_t bh = (uint32_t)(b >> VERIFY_BITWISE_12_LIMB_BITS);

        uint32_t column_index = (ah << VERIFY_BITWISE_12_EXPAND_BITS) + bh;
        uint32_t row_index = (al << VERIFY_BITWISE_12_LIMB_BITS) + bl;
        atomicAdd(&mults[column_index][row_index], 1);
    }
}


void verify_bitwise_xor_12_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);
    m31 **device_mults = clone_to_device<m31*>(mults, mults_col_size);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_12_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        device_mults,
        mults_col_size,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
    cuda_free_memory(device_mults);
}
