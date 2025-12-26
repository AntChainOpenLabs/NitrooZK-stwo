
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
#include "../constraints/relations.cuh"

// Type alias for MemoryAddressToId
typedef LookupElementsBasic<2> MemoryAddressToId;

__global__ void memory_address_to_id_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        // Process all input columns (e.g., 29 for add_mod_builtin)
        for (unsigned col = 0; col < input_col_sizes; col++) {
            m31 value = inputs[col][row];
            // 0 indicates padding row, skip
            if (value != 0) {
                uint32_t addr = (value - 1);
                uint32_t mults_size = 1u << mults_row_log_size;
                if (addr < mults_size) {
                    atomicAdd(&mults[addr], 1);
                }
            }
        }
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

#define MEMORY_ADDRESS_TO_ID_SPLIT 16
#define N_ID_AND_MULT_COLUMNS_PER_CHUNK 2
#define N_TRACE_COLUMNS (MEMORY_ADDRESS_TO_ID_SPLIT * N_ID_AND_MULT_COLUMNS_PER_CHUNK)

__launch_bounds__(256, 2)
__global__ void generate_memory_address_to_id_trace_kernel(
    m31 **traces,
    m31 *address_to_raw_id,
    m31 *multiplicities,
    unsigned total_size_log,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < trace_size) {
        // Calculate which chunk this row belongs to
        unsigned n_packed_rows = trace_size;

        // For each chunk, write ID and multiplicity
        for (unsigned chunk_idx = 0; chunk_idx < MEMORY_ADDRESS_TO_ID_SPLIT; chunk_idx++) {
            unsigned global_row = row + chunk_idx * trace_size;

            // Read ID and multiplicity from input arrays
            m31 id = {0};
            m31 mult = {0};

            if (global_row < (1u << total_size_log)) {
                id = address_to_raw_id[global_row];
                mult = multiplicities[global_row];
            }

            // Write to trace columns
            // Each chunk has 2 columns: [id, multiplicity]
            unsigned id_col = chunk_idx * N_ID_AND_MULT_COLUMNS_PER_CHUNK;
            unsigned mult_col = id_col + 1;

            traces[id_col][row] = id;
            traces[mult_col][row] = mult;
        }
    }
}

__launch_bounds__(256, 2)
__global__ void generate_memory_address_to_id_interaction_trace_kernel(
    m31 **interaction_traces,
    m31 **traces,
    unsigned log_size,
    unsigned trace_size,
    MemoryAddressToId *lookup_elements
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < trace_size) {
        // We process pairs of chunks (0,1), (2,3), ..., (14,15)
        // Each pair produces one interaction column
        unsigned num_pairs = MEMORY_ADDRESS_TO_ID_SPLIT / 2;

        for (unsigned pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
            unsigned chunk0 = pair_idx * 2;
            unsigned chunk1 = chunk0 + 1;

            // Read IDs and multiplicities for both chunks
            m31 id0 = traces[chunk0 * N_ID_AND_MULT_COLUMNS_PER_CHUNK][row];
            m31 mult0 = traces[chunk0 * N_ID_AND_MULT_COLUMNS_PER_CHUNK + 1][row];
            m31 id1 = traces[chunk1 * N_ID_AND_MULT_COLUMNS_PER_CHUNK][row];
            m31 mult1 = traces[chunk1 * N_ID_AND_MULT_COLUMNS_PER_CHUNK + 1][row];

            // Calculate addresses (row + 1 for offset, plus chunk offset)
            m31 addr0 = row + 1 + chunk0 * trace_size;
            m31 addr1 = row + 1 + chunk1 * trace_size;

            // Compute logup values: combine(addr, id) for each
            m31 p0_inputs[2] = {addr0, id0};
            qm31 p0 = lookup_elements->combine(p0_inputs, 2);

            m31 p1_inputs[2] = {addr1, id1};
            qm31 p1 = lookup_elements->combine(p1_inputs, 2);

            // Compute numerator: p0 * (-mult1) + p1 * (-mult0)
            // Convert m31 to qm31: qm31{cm31{m, 0}, cm31{0, 0}}
            qm31 zero = qm31{cm31{0, 0}, cm31{0, 0}};
            qm31 qm31_mult0 = qm31{cm31{mult0, 0}, cm31{0, 0}};
            qm31 qm31_mult1 = qm31{cm31{mult1, 0}, cm31{0, 0}};
            qm31 neg_mult0 = sub(zero, qm31_mult0);
            qm31 neg_mult1 = sub(zero, qm31_mult1);

            qm31 term0 = mul(p0, neg_mult1);
            qm31 term1 = mul(p1, neg_mult0);
            qm31 numerator = add(term0, term1);

            // Compute denominator: p1 * p0
            qm31 denominator = mul(p1, p0);

            // Store numerator and denominator for this pair
            // Each interaction column is for one pair
            unsigned base_col = pair_idx * 8; // 8 M31 values per QM31 fraction (4 for num, 4 for denom)

            // Numerator (4 M31 values)
            interaction_traces[base_col + 0][row] = numerator.a.a;
            interaction_traces[base_col + 1][row] = numerator.a.b;
            interaction_traces[base_col + 2][row] = numerator.b.a;
            interaction_traces[base_col + 3][row] = numerator.b.b;

            // Denominator (4 M31 values)
            interaction_traces[base_col + 4][row] = denominator.a.a;
            interaction_traces[base_col + 5][row] = denominator.a.b;
            interaction_traces[base_col + 6][row] = denominator.b.a;
            interaction_traces[base_col + 7][row] = denominator.b.b;
        }
    }
}

void generate_memory_address_to_id_traces(
    m31 **traces,
    m31 **interaction_traces,
    m31 *address_to_raw_id,
    m31 *multiplicities,
    unsigned total_size_log,
    unsigned log_size,
    MemoryAddressToId *lookup_elements
) {
    unsigned trace_size = 1u << log_size;

    // Copy traces pointer to device
    m31 **device_traces = clone_to_device<m31*>(traces, N_TRACE_COLUMNS);

    // Generate main trace
    int block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    generate_memory_address_to_id_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        address_to_raw_id,
        multiplicities,
        total_size_log,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Generate interaction trace if needed
    if (interaction_traces != nullptr && lookup_elements != nullptr) {
        unsigned num_interaction_cols = 8 * (MEMORY_ADDRESS_TO_ID_SPLIT / 2);
        m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, num_interaction_cols);

        // Copy lookup_elements to device memory
        MemoryAddressToId *device_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
        ASSERT_CUDA_SUCCESS(cudaMemcpy(device_lookup_elements, lookup_elements, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice));

        generate_memory_address_to_id_interaction_trace_kernel<<<num_blocks, block_dim>>>(
            device_interaction_traces,
            device_traces,
            log_size,
            trace_size,
            device_lookup_elements
        );

        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        cuda_free_memory(device_interaction_traces);
        cuda_free_memory(device_lookup_elements);
    }

    cuda_free_memory(device_traces);
}
