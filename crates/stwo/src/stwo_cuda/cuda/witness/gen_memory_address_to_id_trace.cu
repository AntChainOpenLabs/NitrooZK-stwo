
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
#include "gen_memory_address_to_id_trace.cuh"

__global__ void memory_address_to_id_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] - 1);
        atomicAdd(&mults[addr], 1);
    }
}


void memory_address_to_id_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 1 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    memory_address_to_id_add_inputs_kernel<<<num_blocks, block_dim>>>(
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
