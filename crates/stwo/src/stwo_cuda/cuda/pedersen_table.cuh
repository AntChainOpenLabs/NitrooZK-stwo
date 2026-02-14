// CUDA implementation of Pedersen table storage and lookup
// The table stores pre-computed EC points for the Pedersen hash
//
// Table structure (matches CPU implementation):
// - P0 section: 14 blocks × 2^18 rows for value A low bits
// - P1 section: 16 rows for value A high bits
// - P2 section: 14 blocks × 2^18 rows for value B low bits
// - P3 section: 16 rows for value B high bits
//
// Each row contains an (x, y) point where x and y are Felt252 (252-bit field elements)
// stored as 28 M31 limbs each (56 total columns)

#ifndef PEDERSEN_TABLE_CUH
#define PEDERSEN_TABLE_CUH

#include "fields.cuh"
#include "ec_ops.cuh"

// Table parameters (must match Rust constants)
#define PEDERSEN_BITS_PER_WINDOW 18
#define PEDERSEN_NUM_WINDOWS 14  // ceil(252 / 18)
#define PEDERSEN_ROWS_PER_WINDOW (1 << PEDERSEN_BITS_PER_WINDOW)  // 262144

#define PEDERSEN_P0_SECTION_START 0
#define PEDERSEN_P1_SECTION_START (PEDERSEN_P0_SECTION_START + PEDERSEN_NUM_WINDOWS * PEDERSEN_ROWS_PER_WINDOW)
#define PEDERSEN_P2_SECTION_START (PEDERSEN_P1_SECTION_START + 16)
#define PEDERSEN_P3_SECTION_START (PEDERSEN_P2_SECTION_START + PEDERSEN_NUM_WINDOWS * PEDERSEN_ROWS_PER_WINDOW)
#define PEDERSEN_TABLE_N_ROWS_UNPADDED (PEDERSEN_P3_SECTION_START + 16)

// Table column count: 28 M31 limbs for x + 28 M31 limbs for y = 56 columns
#define PEDERSEN_TABLE_N_COLUMNS 56

// Global GPU storage for the Pedersen table
// This is allocated once and reused across all kernel calls
// Note: Definitions are in pedersen_table_init.cu
extern __device__ m31* g_pedersen_table_columns[PEDERSEN_TABLE_N_COLUMNS];
extern __device__ uint32_t g_pedersen_table_n_rows;

// Host-side pointers for table management (defined inline in implementation section below)

// Initialize the Pedersen table on GPU
// Called once from Rust to upload the table
// columns: array of 56 column pointers (each column has n_rows M31 elements)
// n_rows: number of rows in the table (must be power of 2)
extern "C" void pedersen_table_init(m31** columns, uint32_t n_rows);

// Free the Pedersen table from GPU memory
extern "C" void pedersen_table_free();

// Device function to look up a point from the table
// table_row: row index in the table
// x_limbs, y_limbs: output arrays (28 M31 each)
__device__ void pedersen_table_lookup(
    uint32_t table_row,
    m31* x_limbs,
    m31* y_limbs
);

// Device function to reconstruct Felt252 from 28 M31 limbs
// Input: 28 M31 values where each is a 9-bit limb
// Output: 256-bit value in ff_storage<8> format
__device__ void felt252_from_28_limbs(felt252& result, const m31* limbs);

// Device function to split Felt252 into 28 M31 limbs
// Input: 256-bit value in ff_storage<8> format
// Output: 28 M31 values (9 bits each)
__device__ void felt252_to_28_limbs(const felt252& value, m31* limbs);

// ============================================================================
// Implementation
// ============================================================================

// Host-side variables for CPU upload path
inline m31** h_pedersen_table_device_ptrs = nullptr;
inline bool g_pedersen_table_initialized = false;

// Kernel to copy table column pointers to device global variables
static __global__ void set_pedersen_table_pointers_kernel(m31** column_ptrs, uint32_t n_rows) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < PEDERSEN_TABLE_N_COLUMNS; i++) {
            g_pedersen_table_columns[i] = column_ptrs[i];
        }
        g_pedersen_table_n_rows = n_rows;
    }
}

// Internal implementation for CPU upload (used by gen_pedersen_builtin_trace.cu)
inline void pedersen_table_init_impl(m31** columns, uint32_t n_rows) {
    if (g_pedersen_table_initialized) {
        return;  // Already initialized
    }

    // Clone column pointers to device
    h_pedersen_table_device_ptrs = clone_to_device<m31*>(columns, PEDERSEN_TABLE_N_COLUMNS);

    // Set the global device pointers
    set_pedersen_table_pointers_kernel<<<1, 1>>>(h_pedersen_table_device_ptrs, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    g_pedersen_table_initialized = true;
}

inline void pedersen_table_free_impl() {
    if (!g_pedersen_table_initialized) {
        return;
    }

    // Free the pointer array
    if (h_pedersen_table_device_ptrs != nullptr) {
        cuda_free_memory(h_pedersen_table_device_ptrs);
        h_pedersen_table_device_ptrs = nullptr;
    }

    g_pedersen_table_initialized = false;
}

__device__ __forceinline__ void pedersen_table_lookup(
    uint32_t table_row,
    m31* x_limbs,
    m31* y_limbs
) {
    // Read x coordinate (first 28 columns)
    for (int i = 0; i < 28; i++) {
        x_limbs[i] = g_pedersen_table_columns[i][table_row];
    }
    // Read y coordinate (next 28 columns)
    for (int i = 0; i < 28; i++) {
        y_limbs[i] = g_pedersen_table_columns[28 + i][table_row];
    }
}

__device__ __forceinline__ void felt252_from_28_limbs(felt252& result, const m31* limbs) {
    // Clear result
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = 0;
    }

    // Each M31 limb is 9 bits (252 bits total = 28 × 9)
    // We need to combine them into 8 × 32 = 256 bits
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int result_idx = 0;

    for (int i = 0; i < 28; i++) {
        uint64_t value = limbs[i] & 0x1FF;  // 9 bits
        accumulator |= (value << bits_in_acc);
        bits_in_acc += 9;

        // Extract 32-bit chunks when we have enough
        while (bits_in_acc >= 32 && result_idx < 8) {
            result.limbs[result_idx] = (uint32_t)(accumulator & 0xFFFFFFFF);
            accumulator >>= 32;
            bits_in_acc -= 32;
            result_idx++;
        }
    }

    // Handle any remaining bits
    if (result_idx < 8 && accumulator != 0) {
        result.limbs[result_idx] = (uint32_t)accumulator;
    }
}

__device__ __forceinline__ void felt252_to_28_limbs(const felt252& value, m31* limbs) {
    // Extract 9 bits at a time from the 256-bit value
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int limb_idx = 0;

    for (int i = 0; i < 28; i++) {
        // Load more bits if needed
        while (bits_in_acc < 9 && limb_idx < 8) {
            accumulator |= ((uint64_t)value.limbs[limb_idx]) << bits_in_acc;
            bits_in_acc += 32;
            limb_idx++;
        }

        // Extract 9 bits
        limbs[i] = (m31){(uint32_t)(accumulator & 0x1FF)};
        accumulator >>= 9;
        bits_in_acc -= 9;
    }
}

#endif // PEDERSEN_TABLE_CUH
