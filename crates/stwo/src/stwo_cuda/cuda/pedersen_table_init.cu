// CUDA Pedersen Table GPU-Native Initialization
// Similar to evaluate_poseidon_constraint.cu's initialize_poseidon_constants()
//
// This file generates the PEDERSEN_TABLE directly on GPU and stores pointers
// in the existing global device memory symbols (g_pedersen_table_columns).
//
// Key differences from CPU upload approach:
// - Table is generated entirely on GPU (no host->device transfer of ~1.8GB)
// - Uses existing pedersen_table.cuh global symbols
// - Provides faster initialization for repeated test runs

#include "fields.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "fp256_config.cuh"
#include "fp256_dispatch_st.cuh"
#include "cuda_mem_pool.cuh"
#include <cstdint>
#include <cstdio>

// Include ec_ops.cuh for felt252 operations
#include "ec_ops.cuh"

// ============================================================================
// Table Parameters (must match pedersen_table.cuh)
// ============================================================================
#define INIT_PEDERSEN_BITS_PER_WINDOW 18
#define INIT_PEDERSEN_NUM_WINDOWS 14
#define INIT_PEDERSEN_ROWS_PER_WINDOW (1 << INIT_PEDERSEN_BITS_PER_WINDOW)  // 262144

#define INIT_PEDERSEN_P0_START 0
#define INIT_PEDERSEN_P1_START (INIT_PEDERSEN_P0_START + INIT_PEDERSEN_NUM_WINDOWS * INIT_PEDERSEN_ROWS_PER_WINDOW)  // 3670016
#define INIT_PEDERSEN_P2_START (INIT_PEDERSEN_P1_START + 16)  // 3670032
#define INIT_PEDERSEN_P3_START (INIT_PEDERSEN_P2_START + INIT_PEDERSEN_NUM_WINDOWS * INIT_PEDERSEN_ROWS_PER_WINDOW)  // 7340048
#define INIT_PEDERSEN_TABLE_N_ROWS_UNPADDED (INIT_PEDERSEN_P3_START + 16)  // 7340064
#define INIT_PEDERSEN_TABLE_N_COLUMNS 56

// ============================================================================
// Global Device Memory Storage for Pedersen Table
// These are the actual definitions (declared extern in pedersen_table.cuh)
// ============================================================================
__device__ m31* g_pedersen_table_columns[56];
__device__ uint32_t g_pedersen_table_n_rows = 0;

// Host-side tracking for GPU-generated table
static bool s_pedersen_table_gpu_initialized = false;
static m31* s_pedersen_table_gpu_ptrs[INIT_PEDERSEN_TABLE_N_COLUMNS] = {nullptr};
static uint32_t s_pedersen_table_gpu_n_rows = 0;

// ============================================================================
// Starknet Pedersen Curve Constants (stored in __constant__ for fast access)
// ============================================================================

// PEDERSEN_P0 (generator for low 252 bits of value A)
// P0.x = 0x0234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b
// P0.y = 0x03b056f100f96fb21e889527d41f4e39940135dd7a6c94cc6ed0268ee89e5615
__constant__ uint32_t CONST_PEDERSEN_P0_X[8] = {
    0x57ebe47b, 0x1080d179, 0x6d56eb0c, 0x8fa8120b,
    0x55fca9e5, 0x969c7486, 0xcbaffe7f, 0x0234287d
};
__constant__ uint32_t CONST_PEDERSEN_P0_Y[8] = {
    0xe89e5615, 0x6ed0268e, 0x7a6c94cc, 0x940135dd,
    0xd41f4e39, 0x1e889527, 0x00f96fb2, 0x03b056f1
};

// PEDERSEN_P1 (generator for high 4 bits of value A)
// P1.x = 0x04fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378
// P1.y = 0x03fa0984c931c9e38113e0c0e47e4401562761f92a7a23b45168f4e80ff5b54d
__constant__ uint32_t CONST_PEDERSEN_P1_X[8] = {
    0xba8aa378, 0xb7a6932d, 0xde5e3018, 0x99099ec1,
    0x56558f33, 0x3f9dab26, 0x76c83db3, 0x04fa56f3
};
__constant__ uint32_t CONST_PEDERSEN_P1_Y[8] = {
    0x0ff5b54d, 0x5168f4e8, 0x2a7a23b4, 0x562761f9,
    0xe47e4401, 0x8113e0c0, 0xc931c9e3, 0x03fa0984
};

// PEDERSEN_P2 (generator for low 252 bits of value B)
// P2.x = 0x04ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997
// P2.y = 0x0040301cf5c1751f4b971e46c4ede85fcac5c59a5ce5ae7c48151f27b24b219c
__constant__ uint32_t CONST_PEDERSEN_P2_X[8] = {
    0xbd2d6997, 0x3aa372f0, 0x4709e90f, 0x40c690c7,
    0x5b45f74b, 0x764910f7, 0x66be8dec, 0x04ba4cc1
};
__constant__ uint32_t CONST_PEDERSEN_P2_Y[8] = {
    0xb24b219c, 0x48151f27, 0x5ce5ae7c, 0xcac5c59a,
    0xc4ede85f, 0x4b971e46, 0xf5c1751f, 0x0040301c
};

// PEDERSEN_P3 (generator for high 4 bits of value B)
// P3.x = 0x054302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202
// P3.y = 0x01b77b3e37d13504b348046268d8ae25ce98ad783c25561a879dcc77e99c2426
__constant__ uint32_t CONST_PEDERSEN_P3_X[8] = {
    0x49a58202, 0xd36ff12c, 0xd53fb325, 0x2ca65048,
    0xf61a63bb, 0x6e44cca8, 0xb0e6cc1c, 0x054302dc
};
__constant__ uint32_t CONST_PEDERSEN_P3_Y[8] = {
    0xe99c2426, 0x879dcc77, 0x3c25561a, 0xce98ad78,
    0x68d8ae25, 0xb3480462, 0x37d13504, 0x01b77b3e
};

// SHIFT_POINT (negated and added to all table entries)
// shift_point.x = 0x049ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804
// shift_point.y = 0x03ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a
__constant__ uint32_t CONST_SHIFT_POINT_X[8] = {
    0x50ca6804, 0x551fde40, 0x22947733, 0x716b0b10,
    0xeb599f16, 0x00ee1b87, 0xa8c16007, 0x049ee3eb
};
__constant__ uint32_t CONST_SHIFT_POINT_Y[8] = {
    0x6e10268a, 0xd0405d26, 0xc0e056c1, 0x4e621062,
    0x06ea0ed3, 0xf346d49d, 0x4b3bc6dd, 0x03ca0cfe
};

// ============================================================================
// Device Helper Functions
// ============================================================================

// Load base points directly from constant memory (avoids pointer passing issues)
__device__ void load_P0(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = CONST_PEDERSEN_P0_X[i];
        y.limbs[i] = CONST_PEDERSEN_P0_Y[i];
    }
}

__device__ void load_P1(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = CONST_PEDERSEN_P1_X[i];
        y.limbs[i] = CONST_PEDERSEN_P1_Y[i];
    }
}

__device__ void load_P2(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = CONST_PEDERSEN_P2_X[i];
        y.limbs[i] = CONST_PEDERSEN_P2_Y[i];
    }
}

__device__ void load_P3(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = CONST_PEDERSEN_P3_X[i];
        y.limbs[i] = CONST_PEDERSEN_P3_Y[i];
    }
}

__device__ void load_shift_point(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = CONST_SHIFT_POINT_X[i];
        y.limbs[i] = CONST_SHIFT_POINT_Y[i];
    }
}

// Negate Y coordinate of a projective point (in Montgomery form)
__device__ void negate_projective_y(ProjectivePointCuda& P) {
    // In Montgomery form, negation is: -Y = 0 - Y
    // Zero is represented as all zeros in both standard and Montgomery form
    felt252 zero = {};
    for (int i = 0; i < 8; i++) zero.limbs[i] = 0;
    P.Y = felt_sub(zero, P.Y);
}

// Convert felt252 to 28 M31 limbs (9 bits each)
// Using naming that doesn't conflict with ec_ops.cuh
__device__ void felt252_to_m31_28_limbs(const felt252& value, m31* limbs) {
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int limb_idx = 0;

    for (int i = 0; i < 28; i++) {
        while (bits_in_acc < 9 && limb_idx < 8) {
            accumulator |= ((uint64_t)value.limbs[limb_idx]) << bits_in_acc;
            bits_in_acc += 32;
            limb_idx++;
        }
        limbs[i] = (m31){(uint32_t)(accumulator & 0x1FF)};
        accumulator >>= 9;
        bits_in_acc -= 9;
    }
}

// ============================================================================
// EC Point Operations for Table Generation
// ============================================================================

__device__ void ec_double_projective(ProjectivePointCuda& P) {
    felt252 X = P.X;
    felt252 Y = P.Y;
    felt252 Z = P.Z;

    // Formula dbl-2007-bl from https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html
    // For curve y² = x³ + ax + b with a = 1 (Starknet curve)
    // w = 3 * X^2 + a * Z^2
    felt252 XX = felt_mul(X, X);
    felt252 ZZ = felt_mul(Z, Z);
    felt252 w = felt_add(felt_add(XX, XX), XX);  // 3*X^2
    w = felt_add(w, ZZ);  // + Z^2

    // s = 2*Y*Z (FIX: was Y*Z which caused all downstream values to be wrong)
    felt252 YZ = felt_mul(Y, Z);
    felt252 s = felt_add(YZ, YZ);  // s = 2*Y*Z
    felt252 ss = felt_mul(s, s);
    felt252 sss = felt_mul(s, ss);
    felt252 R = felt_mul(Y, s);    // R = Y*s = 2*Y²*Z
    felt252 RR = felt_mul(R, R);   // RR = R² = 4*Y⁴*Z²

    felt252 X_plus_R = felt_add(X, R);
    felt252 B = felt_mul(X_plus_R, X_plus_R);
    B = felt_sub(B, XX);
    B = felt_sub(B, RR);  // B = (X+R)² - X² - R² = 2*X*R = 4*X*Y²*Z

    felt252 ww = felt_mul(w, w);
    felt252 two_B = felt_add(B, B);
    felt252 h = felt_sub(ww, two_B);  // h = w² - 2*B

    P.X = felt_mul(h, s);  // X3 = h*s

    felt252 B_minus_h = felt_sub(B, h);
    felt252 w_Bh = felt_mul(w, B_minus_h);
    felt252 two_RR = felt_add(RR, RR);
    P.Y = felt_sub(w_Bh, two_RR);  // Y3 = w*(B-h) - 2*RR

    // Z3 = s³ (FIX: was 8*s³ which was a hack to compensate for wrong s)
    P.Z = sss;
}

// ============================================================================
// Table Generation Kernels
// ============================================================================

// Point type enum to select which base point to use
enum PedersenPointType {
    POINT_P0 = 0,
    POINT_P1 = 1,
    POINT_P2 = 2,
    POINT_P3 = 3
};

__device__ void load_base_point(PedersenPointType point_type, felt252& x, felt252& y) {
    switch (point_type) {
        case POINT_P0: load_P0(x, y); break;
        case POINT_P1: load_P1(x, y); break;
        case POINT_P2: load_P2(x, y); break;
        case POINT_P3: load_P3(x, y); break;
    }
}

// ============================================================================
// Optimized Kernel using Binary Decomposition
// ============================================================================
//
// Key insight: Entry[k] = -SHIFT_POINT + k * scaled_base
// where k = b_17*2^17 + b_16*2^16 + ... + b_1*2 + b_0 (18 bits)
//
// So Entry[k] = -SHIFT_POINT + sum(b_i * 2^i * scaled_base) for bits where b_i=1
//
// We precompute 18 powers: powers[i] = 2^i * scaled_base
// Each thread only adds powers for set bits (max 18 additions vs 262,143)
//
__global__ void gen_pedersen_block_optimized_kernel(
    m31** columns,
    uint32_t block_start_row,
    uint32_t n_rows_in_block,
    PedersenPointType point_type,
    uint32_t window
) {
    // Shared memory for 18 precomputed powers: powers[i] = 2^i * scaled_base
    __shared__ AffinePointCuda s_powers[INIT_PEDERSEN_BITS_PER_WINDOW];

    // Phase 0: Thread 0 computes the 18 powers
    if (threadIdx.x == 0) {
        // Load base point from constant memory
        AffinePointCuda base;
        load_base_point(point_type, base.x, base.y);

        // Convert to projective (Montgomery form)
        ProjectivePointCuda P = affine_to_projective(base);

        // Scale by 2^(18*window) to get the starting point for this window
        for (uint32_t i = 0; i < INIT_PEDERSEN_BITS_PER_WINDOW * window; i++) {
            ec_double_projective(P);
        }

        // Compute 18 powers: powers[i] = 2^i * scaled_base
        // powers[0] = scaled_base
        // powers[1] = 2 * scaled_base
        // powers[2] = 4 * scaled_base
        // ...
        // powers[17] = 2^17 * scaled_base
        for (int i = 0; i < INIT_PEDERSEN_BITS_PER_WINDOW; i++) {
            projective_to_affine(P, s_powers[i]);
            ec_double_projective(P);
        }
    }
    __syncthreads();

    // Each thread computes its entry using binary decomposition
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_rows_in_block) return;

    // Phase 1: Start with -SHIFT_POINT
    AffinePointCuda shift;
    load_shift_point(shift.x, shift.y);
    ProjectivePointCuda acc = affine_to_projective(shift);
    negate_projective_y(acc);

    // Phase 2: Binary decomposition - add only powers for set bits
    // Entry[k] = -SHIFT_POINT + sum(b_i * powers[i]) where b_i is bit i of k
    #pragma unroll
    for (int bit = 0; bit < INIT_PEDERSEN_BITS_PER_WINDOW; bit++) {
        if (k & (1u << bit)) {
            ec_add_mixed(acc, s_powers[bit]);
        }
    }

    // Phase 3: Convert to affine (includes field inversion)
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Phase 4: Store to columns
    m31 x_limbs[28], y_limbs[28];
    felt252_to_m31_28_limbs(result.x, x_limbs);
    felt252_to_m31_28_limbs(result.y, y_limbs);

    uint32_t output_row = block_start_row + k;
    for (int i = 0; i < 28; i++) {
        columns[i][output_row] = x_limbs[i];
        columns[28 + i][output_row] = y_limbs[i];
    }
}

// Legacy kernel (kept for reference/fallback, but not used)
__global__ void gen_pedersen_block_kernel_init(
    m31** columns,
    uint32_t block_start_row,
    uint32_t n_rows_in_block,
    PedersenPointType point_type,
    uint32_t window
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows_in_block) return;

    uint32_t output_row = block_start_row + idx;

    // Load base point from constant memory
    AffinePointCuda base_point;
    load_base_point(point_type, base_point.x, base_point.y);

    // Compute 2^(18*window) * base_point
    ProjectivePointCuda scaled_base = affine_to_projective(base_point);
    for (uint32_t i = 0; i < 18 * window; i++) {
        ec_double_projective(scaled_base);
    }

    // Load -SHIFT_POINT as initial accumulator
    AffinePointCuda shift_point;
    load_shift_point(shift_point.x, shift_point.y);

    // Convert to projective (which converts to Montgomery form), then negate Y
    ProjectivePointCuda acc = affine_to_projective(shift_point);
    negate_projective_y(acc);

    // Convert scaled_base to affine for mixed addition
    AffinePointCuda scaled_base_affine;
    projective_to_affine(scaled_base, scaled_base_affine);

    // Add scaled_base idx times
    for (uint32_t k = 0; k < idx; k++) {
        ec_add_mixed(acc, scaled_base_affine);
    }

    // Convert to affine
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Convert to 28 M31 limbs and store
    m31 x_limbs[28], y_limbs[28];
    felt252_to_m31_28_limbs(result.x, x_limbs);
    felt252_to_m31_28_limbs(result.y, y_limbs);

    // Store to output columns
    for (int i = 0; i < 28; i++) {
        columns[i][output_row] = x_limbs[i];
        columns[28 + i][output_row] = y_limbs[i];
    }
}

__global__ void gen_pedersen_small_section_kernel_init(
    m31** columns,
    uint32_t section_start_row,
    uint32_t n_rows_in_section,
    PedersenPointType point_type
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows_in_section) return;

    uint32_t output_row = section_start_row + idx;

    // Load base point from constant memory
    AffinePointCuda base_point;
    load_base_point(point_type, base_point.x, base_point.y);

    // Load -SHIFT_POINT as initial accumulator
    AffinePointCuda shift_point;
    load_shift_point(shift_point.x, shift_point.y);

    // Convert to projective (Montgomery form), then negate Y
    ProjectivePointCuda acc = affine_to_projective(shift_point);
    negate_projective_y(acc);

    for (uint32_t k = 0; k < idx; k++) {
        ec_add_mixed(acc, base_point);
    }

    AffinePointCuda result;
    projective_to_affine(acc, result);

    m31 x_limbs[28], y_limbs[28];
    felt252_to_m31_28_limbs(result.x, x_limbs);
    felt252_to_m31_28_limbs(result.y, y_limbs);

    for (int i = 0; i < 28; i++) {
        columns[i][output_row] = x_limbs[i];
        columns[28 + i][output_row] = y_limbs[i];
    }
}

__global__ void set_global_pedersen_table_pointers_kernel(m31** ptrs, uint32_t n_rows) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < INIT_PEDERSEN_TABLE_N_COLUMNS; i++) {
            g_pedersen_table_columns[i] = ptrs[i];
        }
        g_pedersen_table_n_rows = n_rows;
    }
}

// ============================================================================
// External C API - Similar to initialize_poseidon_constants()
// ============================================================================

extern "C" void initialize_pedersen_table() {
    if (s_pedersen_table_gpu_initialized) {
        printf("[PEDERSEN_TABLE_GPU] Already initialized, skipping.\n");
        return;
    }

    timer global_timer;
    global_timer.start("initialize_pedersen_table (GPU generation)");

    // Compute padded size (next power of 2)
    uint32_t n_rows_unpadded = INIT_PEDERSEN_TABLE_N_ROWS_UNPADDED;
    uint32_t n_rows = 1;
    while (n_rows < n_rows_unpadded) n_rows <<= 1;

    s_pedersen_table_gpu_n_rows = n_rows;
    printf("[PEDERSEN_TABLE_GPU] Allocating %u rows × %d columns (%zu MB)...\n",
           n_rows, INIT_PEDERSEN_TABLE_N_COLUMNS,
           (size_t)n_rows * INIT_PEDERSEN_TABLE_N_COLUMNS * sizeof(m31) / (1024 * 1024));

    // Allocate GPU memory for each column
    for (int i = 0; i < INIT_PEDERSEN_TABLE_N_COLUMNS; i++) {
        s_pedersen_table_gpu_ptrs[i] = cuda_malloc<m31>(n_rows);
        cudaMemset(s_pedersen_table_gpu_ptrs[i], 0, n_rows * sizeof(m31));
    }

    // Clone pointers to device
    m31** d_columns = clone_to_device<m31*>(s_pedersen_table_gpu_ptrs, INIT_PEDERSEN_TABLE_N_COLUMNS);

    const uint32_t BLOCK_SIZE = 256;

    // Generate P0 section (14 windows × 262144 rows each)
    // Using optimized binary decomposition kernel: O(log k) additions instead of O(k)
    printf("[PEDERSEN_TABLE_GPU] Generating P0 section (optimized binary decomposition)...\n");
    for (uint32_t window = 0; window < INIT_PEDERSEN_NUM_WINDOWS; window++) {
        uint32_t block_start = INIT_PEDERSEN_P0_START + window * INIT_PEDERSEN_ROWS_PER_WINDOW;
        uint32_t num_blocks = (INIT_PEDERSEN_ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_block_optimized_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, block_start, INIT_PEDERSEN_ROWS_PER_WINDOW,
            POINT_P0, window
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // Generate P1 section (16 rows)
    printf("[PEDERSEN_TABLE_GPU] Generating P1 section...\n");
    gen_pedersen_small_section_kernel_init<<<1, 16>>>(
        d_columns, INIT_PEDERSEN_P1_START, 16, POINT_P1
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // Generate P2 section (14 windows × 262144 rows each)
    // Using optimized binary decomposition kernel: O(log k) additions instead of O(k)
    printf("[PEDERSEN_TABLE_GPU] Generating P2 section (optimized binary decomposition)...\n");
    for (uint32_t window = 0; window < INIT_PEDERSEN_NUM_WINDOWS; window++) {
        uint32_t block_start = INIT_PEDERSEN_P2_START + window * INIT_PEDERSEN_ROWS_PER_WINDOW;
        uint32_t num_blocks = (INIT_PEDERSEN_ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_block_optimized_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, block_start, INIT_PEDERSEN_ROWS_PER_WINDOW,
            POINT_P2, window
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // Generate P3 section (16 rows)
    printf("[PEDERSEN_TABLE_GPU] Generating P3 section...\n");
    gen_pedersen_small_section_kernel_init<<<1, 16>>>(
        d_columns, INIT_PEDERSEN_P3_START, 16, POINT_P3
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // Set global device symbol pointers (same symbols used by pedersen_table.cuh)
    set_global_pedersen_table_pointers_kernel<<<1, 1>>>(d_columns, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    cuda_free_memory(d_columns);

    s_pedersen_table_gpu_initialized = true;
    global_timer.end("initialize_pedersen_table (GPU generation)");
    printf("[PEDERSEN_TABLE_GPU] Initialization complete!\n");
}

extern "C" bool is_pedersen_table_initialized() {
    return s_pedersen_table_gpu_initialized;
}

// Get the device pointers and row count for the pedersen table columns.
// Table must be initialized first (via initialize_pedersen_table).
extern "C" void get_pedersen_table_column_ptrs(
    m31** output_ptrs,     // Output: 56 device pointers (host-side array)
    uint32_t* out_n_rows   // Output: padded row count
) {
    for (int i = 0; i < INIT_PEDERSEN_TABLE_N_COLUMNS; i++) {
        output_ptrs[i] = s_pedersen_table_gpu_ptrs[i];
    }
    *out_n_rows = s_pedersen_table_gpu_n_rows;
}

extern "C" void free_pedersen_table() {
    if (!s_pedersen_table_gpu_initialized) {
        return;
    }

    for (int i = 0; i < INIT_PEDERSEN_TABLE_N_COLUMNS; i++) {
        if (s_pedersen_table_gpu_ptrs[i] != nullptr) {
            cuda_free_memory(s_pedersen_table_gpu_ptrs[i]);
            s_pedersen_table_gpu_ptrs[i] = nullptr;
        }
    }

    s_pedersen_table_gpu_initialized = false;
    s_pedersen_table_gpu_n_rows = 0;
    printf("[PEDERSEN_TABLE_GPU] Freed GPU memory.\n");
}

// Upload pedersen table from host (CPU) data.
// This copies pre-computed CPU table columns to GPU, guaranteeing exact match
// with the SIMD reference path. Each column is n_rows M31 values.
extern "C" void upload_pedersen_table_from_host(
    const uint32_t** host_columns,  // 56 host-side column arrays (each n_rows uint32_t)
    uint32_t n_cols,                // number of columns (must be 56)
    uint32_t n_rows                 // padded row count (must be power of 2)
) {
    if (s_pedersen_table_gpu_initialized) {
        printf("[PEDERSEN_TABLE_GPU] Already initialized, skipping host upload.\n");
        return;
    }

    timer global_timer;
    global_timer.start("upload_pedersen_table_from_host");

    s_pedersen_table_gpu_n_rows = n_rows;

    printf("[PEDERSEN_TABLE_GPU] Uploading %u columns × %u rows from host (%zu MB)...\n",
           n_cols, n_rows,
           (size_t)n_rows * n_cols * sizeof(m31) / (1024 * 1024));

    // Allocate GPU memory and copy each column from host
    for (int i = 0; i < INIT_PEDERSEN_TABLE_N_COLUMNS; i++) {
        s_pedersen_table_gpu_ptrs[i] = cuda_malloc<m31>(n_rows);
        if (i < (int)n_cols && host_columns[i] != nullptr) {
            ASSERT_CUDA_SUCCESS(cudaMemcpy(
                s_pedersen_table_gpu_ptrs[i], host_columns[i],
                n_rows * sizeof(m31), cudaMemcpyHostToDevice));
        } else {
            cudaMemset(s_pedersen_table_gpu_ptrs[i], 0, n_rows * sizeof(m31));
        }
    }

    // Set global device symbol pointers
    m31** d_columns = clone_to_device<m31*>(s_pedersen_table_gpu_ptrs, INIT_PEDERSEN_TABLE_N_COLUMNS);
    set_global_pedersen_table_pointers_kernel<<<1, 1>>>(d_columns, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_columns);

    s_pedersen_table_gpu_initialized = true;
    global_timer.end("upload_pedersen_table_from_host");
    printf("[PEDERSEN_TABLE_GPU] Host upload complete!\n");
}

// Debug function: get base point P0 directly from constant memory
__global__ void debug_get_P0_kernel(uint32_t* x_out, uint32_t* y_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            x_out[i] = CONST_PEDERSEN_P0_X[i];
            y_out[i] = CONST_PEDERSEN_P0_Y[i];
        }
    }
}

extern "C" void debug_get_P0_constant(uint32_t* x_limbs, uint32_t* y_limbs) {
    uint32_t* d_x = cuda_mem_pool_allocate<uint32_t>(8);
    uint32_t* d_y = cuda_mem_pool_allocate<uint32_t>(8);

    debug_get_P0_kernel<<<1, 1>>>(d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(x_limbs, d_x, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_limbs, d_y, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cuda_mem_pool_free(d_x);
    cuda_mem_pool_free(d_y);
}

// Debug function: get SHIFT_POINT directly from constant memory
__global__ void debug_get_shift_kernel(uint32_t* x_out, uint32_t* y_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            x_out[i] = CONST_SHIFT_POINT_X[i];
            y_out[i] = CONST_SHIFT_POINT_Y[i];
        }
    }
}

extern "C" void debug_get_shift_constant(uint32_t* x_limbs, uint32_t* y_limbs) {
    uint32_t* d_x = cuda_mem_pool_allocate<uint32_t>(8);
    uint32_t* d_y = cuda_mem_pool_allocate<uint32_t>(8);

    debug_get_shift_kernel<<<1, 1>>>(d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(x_limbs, d_x, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_limbs, d_y, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cuda_mem_pool_free(d_x);
    cuda_mem_pool_free(d_y);
}

// Debug function: test negation of Y coordinate
// Return -SHIFT_POINT in affine form (x unchanged, y negated)
__global__ void debug_negate_shift_kernel(uint32_t* x_out, uint32_t* y_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Load SHIFT_POINT
    AffinePointCuda shift_point;
    load_shift_point(shift_point.x, shift_point.y);

    // Convert to projective (and Montgomery form)
    ProjectivePointCuda proj = affine_to_projective(shift_point);

    // Negate Y
    negate_projective_y(proj);

    // Convert back to affine
    AffinePointCuda result;
    projective_to_affine(proj, result);

    for (int i = 0; i < 8; i++) {
        x_out[i] = result.x.limbs[i];
        y_out[i] = result.y.limbs[i];
    }
}

extern "C" void debug_negate_shift(uint32_t* x_limbs, uint32_t* y_limbs) {
    uint32_t* d_x = cuda_mem_pool_allocate<uint32_t>(8);
    uint32_t* d_y = cuda_mem_pool_allocate<uint32_t>(8);

    debug_negate_shift_kernel<<<1, 1>>>(d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(x_limbs, d_x, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_limbs, d_y, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cuda_mem_pool_free(d_x);
    cuda_mem_pool_free(d_y);
}

// Debug: test to_mont, from_mont, and inverse for P0.x
__global__ void debug_mont_roundtrip_kernel(uint32_t* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    AffinePointCuda p0;
    load_P0(p0.x, p0.y);

    printf("P0.x standard: %08x %08x %08x %08x ...\n",
           p0.x.limbs[0], p0.x.limbs[1], p0.x.limbs[2], p0.x.limbs[3]);

    felt252 mont = felt_to_mont(p0.x);
    printf("P0.x mont:     %08x %08x %08x %08x ...\n",
           mont.limbs[0], mont.limbs[1], mont.limbs[2], mont.limbs[3]);

    felt252 back = felt_from_mont(mont);
    printf("P0.x back:     %08x %08x %08x %08x ...\n",
           back.limbs[0], back.limbs[1], back.limbs[2], back.limbs[3]);

    // Test multiplication: mont(a) * mont(1) should equal mont(a)
    felt252 one_mont = ff_config_starknet::one;
    printf("one_mont:      %08x %08x %08x %08x ...\n",
           one_mont.limbs[0], one_mont.limbs[1], one_mont.limbs[2], one_mont.limbs[3]);

    felt252 mul_one = felt_mul(mont, one_mont);
    printf("mont*one_mont: %08x %08x %08x %08x ...\n",
           mul_one.limbs[0], mul_one.limbs[1], mul_one.limbs[2], mul_one.limbs[3]);

    // Test: a^2 in Montgomery form
    felt252 sq = felt_mul(mont, mont);
    printf("mont^2:        %08x %08x %08x %08x ...\n",
           sq.limbs[0], sq.limbs[1], sq.limbs[2], sq.limbs[3]);

    // Convert back to standard
    felt252 sq_std = felt_from_mont(sq);
    printf("sq_std:        %08x %08x %08x %08x ...\n",
           sq_std.limbs[0], sq_std.limbs[1], sq_std.limbs[2], sq_std.limbs[3]);

    // Test inverse: a * a^(-1) should equal 1
    printf("\n--- Testing inverse ---\n");
    felt252 mont_inv = felt_inverse(mont);
    printf("mont^(-1):     %08x %08x %08x %08x ...\n",
           mont_inv.limbs[0], mont_inv.limbs[1], mont_inv.limbs[2], mont_inv.limbs[3]);

    felt252 product = felt_mul(mont, mont_inv);
    printf("mont*mont^(-1):%08x %08x %08x %08x ...\n",
           product.limbs[0], product.limbs[1], product.limbs[2], product.limbs[3]);
    printf("Expected 1_mont: %08x %08x %08x %08x ...\n",
           one_mont.limbs[0], one_mont.limbs[1], one_mont.limbs[2], one_mont.limbs[3]);

    bool inv_ok = true;
    for (int i = 0; i < 8; i++) {
        if (product.limbs[i] != one_mont.limbs[i]) inv_ok = false;
    }
    printf("Inverse test: %s\n", inv_ok ? "PASS" : "FAIL");

    // Copy result for verification
    for (int i = 0; i < 8; i++) {
        result[i] = sq_std.limbs[i];
    }
}

extern "C" void debug_mont_roundtrip(uint32_t* result) {
    uint32_t* d_result = cuda_mem_pool_allocate<uint32_t>(8);

    debug_mont_roundtrip_kernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda_mem_pool_free(d_result);
}

// Debug: Test -SHIFT + SHIFT = O (point at infinity)
__global__ void debug_shift_plus_neg_shift_kernel(uint32_t* z_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Load SHIFT_POINT twice
    AffinePointCuda shift;
    load_shift_point(shift.x, shift.y);

    // acc = -SHIFT
    ProjectivePointCuda acc = affine_to_projective(shift);
    negate_projective_y(acc);

    printf("Before adding SHIFT to -SHIFT:\n");
    printf("  acc.Z: %08x %08x %08x %08x\n", acc.Z.limbs[0], acc.Z.limbs[1], acc.Z.limbs[2], acc.Z.limbs[3]);

    // Add SHIFT to acc
    ec_add_mixed(acc, shift, true);

    printf("After adding SHIFT:\n");
    printf("  acc.X: %08x %08x %08x %08x\n", acc.X.limbs[0], acc.X.limbs[1], acc.X.limbs[2], acc.X.limbs[3]);
    printf("  acc.Y: %08x %08x %08x %08x\n", acc.Y.limbs[0], acc.Y.limbs[1], acc.Y.limbs[2], acc.Y.limbs[3]);
    printf("  acc.Z: %08x %08x %08x %08x\n", acc.Z.limbs[0], acc.Z.limbs[1], acc.Z.limbs[2], acc.Z.limbs[3]);

    // If Z = 0, we have the point at infinity
    for (int i = 0; i < 8; i++) z_out[i] = acc.Z.limbs[i];
}

extern "C" void debug_shift_plus_neg_shift(uint32_t* z_limbs) {
    uint32_t* d_z = cuda_mem_pool_allocate<uint32_t>(8);
    debug_shift_plus_neg_shift_kernel<<<1, 1>>>(d_z);
    cudaDeviceSynchronize();
    cudaMemcpy(z_limbs, d_z, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda_mem_pool_free(d_z);
}

// Debug kernel: compute -SHIFT + P0 using simple affine addition
__global__ void debug_compute_affine_add_kernel(uint32_t* result_x, uint32_t* result_y) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Create -SHIFT in affine form
    AffinePointCuda neg_shift;
    load_shift_point(neg_shift.x, neg_shift.y);
    // Negate Y coordinate (standard form): -y = p - y
    felt252 zero = {};
    for (int i = 0; i < 8; i++) zero.limbs[i] = 0;
    felt252 neg_y_mont = felt_sub(zero, felt_to_mont(neg_shift.y));
    neg_shift.y = felt_from_mont(neg_y_mont);

    printf("Using AFFINE addition:\n");
    printf("-SHIFT.x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           neg_shift.x.limbs[0], neg_shift.x.limbs[1], neg_shift.x.limbs[2], neg_shift.x.limbs[3],
           neg_shift.x.limbs[4], neg_shift.x.limbs[5], neg_shift.x.limbs[6], neg_shift.x.limbs[7]);
    printf("-SHIFT.y: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           neg_shift.y.limbs[0], neg_shift.y.limbs[1], neg_shift.y.limbs[2], neg_shift.y.limbs[3],
           neg_shift.y.limbs[4], neg_shift.y.limbs[5], neg_shift.y.limbs[6], neg_shift.y.limbs[7]);

    // Load P0
    AffinePointCuda p0;
    load_P0(p0.x, p0.y);

    printf("P0.x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           p0.x.limbs[0], p0.x.limbs[1], p0.x.limbs[2], p0.x.limbs[3],
           p0.x.limbs[4], p0.x.limbs[5], p0.x.limbs[6], p0.x.limbs[7]);
    printf("P0.y: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           p0.y.limbs[0], p0.y.limbs[1], p0.y.limbs[2], p0.y.limbs[3],
           p0.y.limbs[4], p0.y.limbs[5], p0.y.limbs[6], p0.y.limbs[7]);

    // Compute -SHIFT + P0 using affine addition
    AffinePointCuda result;
    ec_add_affine(neg_shift, p0, result, true);

    printf("Result.x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           result.x.limbs[0], result.x.limbs[1], result.x.limbs[2], result.x.limbs[3],
           result.x.limbs[4], result.x.limbs[5], result.x.limbs[6], result.x.limbs[7]);
    printf("Result.y: %08x %08x %08x %08x %08x %08x %08x %08x\n",
           result.y.limbs[0], result.y.limbs[1], result.y.limbs[2], result.y.limbs[3],
           result.y.limbs[4], result.y.limbs[5], result.y.limbs[6], result.y.limbs[7]);

    for (int i = 0; i < 8; i++) {
        result_x[i] = result.x.limbs[i];
        result_y[i] = result.y.limbs[i];
    }
}

extern "C" void debug_compute_affine_add(uint32_t* x_limbs, uint32_t* y_limbs) {
    uint32_t* d_x = cuda_mem_pool_allocate<uint32_t>(8);
    uint32_t* d_y = cuda_mem_pool_allocate<uint32_t>(8);

    debug_compute_affine_add_kernel<<<1, 1>>>(d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(x_limbs, d_x, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_limbs, d_y, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cuda_mem_pool_free(d_x);
    cuda_mem_pool_free(d_y);
}

// Debug kernel: compute -SHIFT + P0 and return the result with verbose tracing
__global__ void debug_compute_shift_plus_p0_kernel(uint32_t* result_x, uint32_t* result_y) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Load SHIFT_POINT
    AffinePointCuda shift_point;
    load_shift_point(shift_point.x, shift_point.y);

    // Convert to projective and negate Y
    ProjectivePointCuda acc = affine_to_projective(shift_point);

    printf("After affine_to_projective (SHIFT in Mont form):\n");
    printf("  X[0-3]: %08x %08x %08x %08x\n", acc.X.limbs[0], acc.X.limbs[1], acc.X.limbs[2], acc.X.limbs[3]);
    printf("  Y[0-3]: %08x %08x %08x %08x\n", acc.Y.limbs[0], acc.Y.limbs[1], acc.Y.limbs[2], acc.Y.limbs[3]);
    printf("  Z[0-3]: %08x %08x %08x %08x\n", acc.Z.limbs[0], acc.Z.limbs[1], acc.Z.limbs[2], acc.Z.limbs[3]);

    negate_projective_y(acc);

    printf("After negate (-SHIFT in Mont form):\n");
    printf("  Y[0-3]: %08x %08x %08x %08x\n", acc.Y.limbs[0], acc.Y.limbs[1], acc.Y.limbs[2], acc.Y.limbs[3]);

    // Load P0
    AffinePointCuda p0;
    load_P0(p0.x, p0.y);

    printf("P0 in standard form:\n");
    printf("  x[0-3]: %08x %08x %08x %08x\n", p0.x.limbs[0], p0.x.limbs[1], p0.x.limbs[2], p0.x.limbs[3]);
    printf("  y[0-3]: %08x %08x %08x %08x\n", p0.y.limbs[0], p0.y.limbs[1], p0.y.limbs[2], p0.y.limbs[3]);

    // Add P0 to accumulator: acc = -SHIFT + P0
    ec_add_mixed(acc, p0, true);  // Enable debug output

    printf("After ec_add_mixed (still in Mont projective):\n");
    printf("  X[0-3]: %08x %08x %08x %08x\n", acc.X.limbs[0], acc.X.limbs[1], acc.X.limbs[2], acc.X.limbs[3]);
    printf("  Y[0-3]: %08x %08x %08x %08x\n", acc.Y.limbs[0], acc.Y.limbs[1], acc.Y.limbs[2], acc.Y.limbs[3]);
    printf("  Z[0-3]: %08x %08x %08x %08x\n", acc.Z.limbs[0], acc.Z.limbs[1], acc.Z.limbs[2], acc.Z.limbs[3]);

    // Convert to affine
    AffinePointCuda result;
    projective_to_affine(acc, result, true);  // Enable debug

    // Output the result as 8 x 32-bit limbs
    for (int i = 0; i < 8; i++) {
        result_x[i] = result.x.limbs[i];
        result_y[i] = result.y.limbs[i];
    }
}

extern "C" void debug_compute_shift_plus_p0(uint32_t* x_limbs, uint32_t* y_limbs) {
    uint32_t* d_x = cuda_mem_pool_allocate<uint32_t>(8);
    uint32_t* d_y = cuda_mem_pool_allocate<uint32_t>(8);

    debug_compute_shift_plus_p0_kernel<<<1, 1>>>(d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(x_limbs, d_x, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_limbs, d_y, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cuda_mem_pool_free(d_x);
    cuda_mem_pool_free(d_y);
}

// Debug function: download specific table entry for comparison
extern "C" void debug_get_pedersen_table_entry(uint32_t row, uint32_t* x_limbs, uint32_t* y_limbs) {
    if (!s_pedersen_table_gpu_initialized) {
        printf("[PEDERSEN_TABLE_GPU] WARNING: Table not initialized!\n");
        return;
    }
    if (row >= s_pedersen_table_gpu_n_rows) {
        printf("[PEDERSEN_TABLE_GPU] WARNING: Row %u >= max %u\n", row, s_pedersen_table_gpu_n_rows);
        return;
    }

    // Download 28 x-coordinate limbs and 28 y-coordinate limbs
    for (int i = 0; i < 28; i++) {
        m31 val;
        cudaMemcpy(&val, &s_pedersen_table_gpu_ptrs[i][row], sizeof(m31), cudaMemcpyDeviceToHost);
        x_limbs[i] = val;
    }
    for (int i = 0; i < 28; i++) {
        m31 val;
        cudaMemcpy(&val, &s_pedersen_table_gpu_ptrs[28 + i][row], sizeof(m31), cudaMemcpyDeviceToHost);
        y_limbs[i] = val;
    }
}
