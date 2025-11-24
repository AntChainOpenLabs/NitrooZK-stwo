#ifndef GEN_RANGE_CHECK_VECTOR_TRACE_H
#define GEN_RANGE_CHECK_VECTOR_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

extern "C"
void range_check_vector_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    unsigned n_range,
    unsigned *ranges,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void partition_into_bit_segments_cuda(
    // unsigned *inputs,
    unsigned total_size,
    unsigned n_range,
    unsigned *n_bits_per_segments,
    unsigned **output_value
);

extern "C"
void range_check_vector_generate_interaction_trace(
    // uint32_t *inputs,
    uint32_t **cols_tmp_vec,

    unsigned n_range,
    unsigned *ranges,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);


#define HANDLE_RANGE_CHECK(TYPE, TEMPLATE_N, CONDITION) \
    if (CONDITION) { \
        TYPE *range_check_lookup_elements = (TYPE *)lookup_element_ptr; \
        TYPE *device_range_check_lookup_elements = cuda_malloc<TYPE>(1); \
        generate_range_check_interaction_trace_col_single_gen_kernel<TEMPLATE_N><<<num_blocks, block_dim>>>( \
            device_range_check_lookup_elements, \
            device_cols_tmp_vec, \
            trace_size, \
            device_logup_denom, \
            device_numerator0, \
            device_numerator1, \
            device_numerator2, \
            device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize()); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        cuda_free_memory(device_range_check_lookup_elements); \
    }

#endif // GEN_RANGE_CHECK_VECTOR_TRACE_H