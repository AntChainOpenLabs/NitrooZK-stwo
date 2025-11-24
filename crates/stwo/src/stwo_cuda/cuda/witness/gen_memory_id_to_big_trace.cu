#include "relations.cuh"

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

#include "gen_memory_id_to_big_trace.cuh"

__global__ void memory_id_to_big_deduce_kernel(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
) {
    unsigned out_values[8];
    EncodedMemoryValueId emv;
    emv.encoded = id;

    MemoryValueId mv = decode_memory_value_id(&emv);

    for (int j = 0; j < 8; ++j) {
        switch (mv.tag) {
            case MEMORY_VALUE_ID_F252: {
                out_values[j] = transpose_big_value_ptr[j][mv.value];
                // printf("MEMORY_VALUE_ID_F252: out_values[j] %u\n", out_values[j]);
                break;
            }
            case MEMORY_VALUE_ID_SMALL: {
                if (j >= 4) {
                    out_values[j] = 0;
                } else {
                    uint32_t limbs[4];
                    u128_to_4_limbs(((u128 *)small_value_ptr)[mv.value], limbs);
                    out_values[j] = limbs[j];
                }
                break;
            }
            case MEMORY_VALUE_ID_EMPTY: {
                printf("Attempted deduce_output on empty memory cell.\\n");
                return;
            }
            default: {
                printf("Invalid MemoryValueId tag: %d\\n", mv.tag);
                return;
            }
        }
    }

    printf("value: ");
    for (int i = 0; i < 8; i++) {
           printf("%d ", out_values[i]);
    }
    printf("\n");

    split_f252(out_values, felt252_out);
}


void memory_id_to_big_deduce_finese_cuda(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
) {

    unsigned **device_transpose_big_value_ptr = clone_to_device<m31 *>(transpose_big_value_ptr, 8);

    // m31 *felt252_out_dev = clone_to_device<m31 *>(felt252_out, 1);
    printf("run memory_id_to_big_deduce_finese_cuda\n");


    memory_id_to_big_deduce_kernel<<<1, 1>>>(
        device_transpose_big_value_ptr,
        small_value_ptr,
        id,
        felt252_out
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());


    cuda_free_memory(device_transpose_big_value_ptr);
    // cuda_free_memory(felt252_out_dev);
}



__global__ void memory_id_to_big_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *big_mults,
    unsigned big_mults_row_log_size,
    m31 *small_mults,
    unsigned small_mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        EncodedMemoryValueId emv;
        emv.encoded = inputs[0][row];
        MemoryValueId mv = decode_memory_value_id(&emv);

        switch (mv.tag) {
            case MEMORY_VALUE_ID_F252: {
                atomicAdd(&big_mults[mv.tag], 1);
                // printf("MEMORY_VALUE_ID_F252: out_values[j] %u\n", out_values[j]);
                break;
            }
            case MEMORY_VALUE_ID_SMALL: {
                atomicAdd(&small_mults[mv.tag], 1);
                break;
            }
            case MEMORY_VALUE_ID_EMPTY: {
                printf("Attempted deduce_output on empty memory cell.\\n");
                return;
            }
            default: {
                printf("Invalid MemoryValueId tag: %d\\n", mv.tag);
                return;
            }
        }
    }
}


void memory_id_to_big_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *big_mults,
    unsigned big_mults_row_log_size,
    m31 *small_mults,
    unsigned small_mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 1 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    memory_id_to_big_add_inputs_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        big_mults,
        big_mults_row_log_size,
        small_mults,
        small_mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}
