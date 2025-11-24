// CUDA implementation of Poseidon hash for Starknet field
// Based on C implementation logic with CUDA field arithmetic

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "fp256_config.cuh"
#include "fp256_dispatch_st.cuh"
#include "fp256_carry_chain.cuh"
#include "ptx.cuh"

// Constants for Poseidon
#define STATE_SIZE 3
#define FULL_ROUNDS 8
#define PARTIAL_ROUNDS 83
#define NUM_CONSTANTS 107

// Include the Poseidon constants
#include "poseidon252_constants.cuh"
#include "poseidon252.cuh"
// Use project-wide CUDA utils/memory helpers
#include "utils.cuh"

// Forward declaration not needed - function defined later in this file

// Device function for field addition using Starknet config
__device__ void field_add_device(ff_storage<8>* result, const ff_storage<8>* a, const ff_storage<8>* b) {
    *result = ff_dispatch_st<ff_config_starknet>::add(*a, *b);
}

// Device function for field subtraction using Starknet config
__device__ void field_sub_device(ff_storage<8>* result, const ff_storage<8>* a, const ff_storage<8>* b) {
    *result = ff_dispatch_st<ff_config_starknet>::sub(*a, *b);
}

// Device function for field multiplication assuming Montgomery inputs
__device__ void field_mul_device(ff_storage<8>* result, const ff_storage<8>* a, const ff_storage<8>* b) {
    *result = ff_dispatch_st<ff_config_starknet>::mul(*a, *b);
}

// Device function for field cubing (x^3) assuming Montgomery inputs
__device__ void field_pow3_device(ff_storage<8>* result, const ff_storage<8>* a) {
    ff_storage<8> a_squared = ff_dispatch_st<ff_config_starknet>::mul(*a, *a);
    *result = ff_dispatch_st<ff_config_starknet>::mul(a_squared, *a);
}

// Device function for mix operation (MDS matrix multiplication)
// M = ((3,1,1), (1,-1,1), (1,1,-3))
__device__ void mix_device(ff_storage<8> state[STATE_SIZE]) {
    ff_storage<8> t, temp;
    ff_storage<8> new_state[STATE_SIZE];

    // t = state[0] + state[1] + state[2]
    field_add_device(&t, &state[0], &state[1]);
    field_add_device(&t, &t, &state[2]);

    // new_state[0] = t + 2*state[0] = 3*state[0] + state[1] + state[2]
    field_add_device(&temp, &state[0], &state[0]);  // 2*state[0]
    field_add_device(&new_state[0], &t, &temp);

    // new_state[1] = t - 2*state[1] = state[0] - state[1] + state[2]
    field_add_device(&temp, &state[1], &state[1]);  // 2*state[1]
    field_sub_device(&new_state[1], &t, &temp);

    // new_state[2] = t - 3*state[2] + state[2] = state[0] + state[1] - 2*state[2]
    // First compute 3*state[2]
    field_add_device(&temp, &state[2], &state[2]);  // 2*state[2]
    field_add_device(&temp, &temp, &state[2]);       // 3*state[2]
    field_sub_device(&new_state[2], &t, &temp);

    // Copy new_state back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = new_state[i];
    }
}

// Device function for Poseidon permutation
__device__ void poseidon_permute_comp_device(ff_storage<8> state[STATE_SIZE]) {
    ff_storage<8> state_mont[STATE_SIZE];

    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        state_mont[i] = ff_dispatch_st<ff_config_starknet>::to_montgomery(state[i]);
    }

    // First half of full rounds (4 rounds)
    for (int i = 0; i < FULL_ROUNDS / 2; i++) {
        // Add round constants to all elements
        field_add_device(&state_mont[0], &state_mont[0], &POSEIDON252_EXTERNAL_ROUND_CONSTS[i][0]);
        field_add_device(&state_mont[1], &state_mont[1], &POSEIDON252_EXTERNAL_ROUND_CONSTS[i][1]);
        field_add_device(&state_mont[2], &state_mont[2], &POSEIDON252_EXTERNAL_ROUND_CONSTS[i][2]);

        // S-box: cube each element
        field_pow3_device(&state_mont[0], &state_mont[0]);
        field_pow3_device(&state_mont[1], &state_mont[1]);
        field_pow3_device(&state_mont[2], &state_mont[2]);

        // MDS matrix multiplication
        mix_device(state_mont);
    }

    // Partial rounds (83 rounds)
    for (int i = 0; i < PARTIAL_ROUNDS; i++) {
        // Add round constant only to state[2]
        field_add_device(&state_mont[2], &state_mont[2], &POSEIDON252_INTERNAL_ROUND_CONSTS[i]);

        // S-box only on state[2]
        field_pow3_device(&state_mont[2], &state_mont[2]);

        // MDS matrix multiplication
        mix_device(state_mont);
    }

    // Second half of full rounds (4 rounds)
    for (int i = 0; i < FULL_ROUNDS / 2; i++) {
        // Add round constants to all elements
        int external_idx = FULL_ROUNDS / 2 + i;
        field_add_device(&state_mont[0], &state_mont[0], &POSEIDON252_EXTERNAL_ROUND_CONSTS[external_idx][0]);
        field_add_device(&state_mont[1], &state_mont[1], &POSEIDON252_EXTERNAL_ROUND_CONSTS[external_idx][1]);
        field_add_device(&state_mont[2], &state_mont[2], &POSEIDON252_EXTERNAL_ROUND_CONSTS[external_idx][2]);

        // S-box: cube each element
        field_pow3_device(&state_mont[0], &state_mont[0]);
        field_pow3_device(&state_mont[1], &state_mont[1]);
        field_pow3_device(&state_mont[2], &state_mont[2]);

        // MDS matrix multiplication
        mix_device(state_mont);
    }

    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = ff_dispatch_st<ff_config_starknet>::from_montgomery(state_mont[i]);
    }
}

// Memory management functions
extern "C" Poseidon252Hash* cuda_malloc_poseidon252_hash(size_t size) {
    // Allocate via custom allocator (uses CUDA mem pool under the hood)
    Poseidon252Hash* ptr = cuda_malloc<Poseidon252Hash>(static_cast<unsigned int>(size));
    return ptr;
}

extern "C" Poseidon252Hash* cuda_alloc_zeroes_poseidon252_hash(size_t size) {
    Poseidon252Hash* ptr = cuda_malloc<Poseidon252Hash>(static_cast<unsigned int>(size));
    cudaMemset(ptr, 0, size * sizeof(Poseidon252Hash));
    return ptr;
}

// Copy functions
extern "C" Poseidon252Hash* copy_poseidon252_hash_vec_from_host_to_device(
    const Poseidon252Hash* from,
    size_t size
) {
    Poseidon252Hash* device_ptr = cuda_malloc<Poseidon252Hash>(static_cast<unsigned int>(size));
    cudaMemcpy(device_ptr, from, size * sizeof(Poseidon252Hash), cudaMemcpyHostToDevice);
    return device_ptr;
}

extern "C" void copy_poseidon252_hash_vec_from_device_to_host(
    const Poseidon252Hash* from,
    Poseidon252Hash* to,
    size_t size
) {
    cudaMemcpy(to, from, size * sizeof(Poseidon252Hash), cudaMemcpyDeviceToHost);
}

extern "C" void copy_poseidon252_hash_vec_from_device_to_device(
    const Poseidon252Hash* from,
    Poseidon252Hash* dst,
    size_t size
) {
    cudaMemcpy(dst, from, size * sizeof(Poseidon252Hash), cudaMemcpyDeviceToDevice);
}

extern "C" void cuda_get_poseidon252_hash(
    const Poseidon252Hash* device_ptr,
    Poseidon252Hash* host_ptr,
    size_t index
) {
    cudaMemcpy(host_ptr, device_ptr + index, sizeof(Poseidon252Hash), cudaMemcpyDeviceToHost);
}

extern "C" void cuda_set_poseidon252_hash(
    Poseidon252Hash* device_ptr,
    size_t index,
    const Poseidon252Hash* host_ptr
) {
    cudaMemcpy(device_ptr + index, host_ptr, sizeof(Poseidon252Hash), cudaMemcpyHostToDevice);
}

// CUDA kernel for commit on first layer (streaming, no device malloc)
__global__ void poseidon252_commit_on_first_layer_kernel(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    uint8_t* result  // result as bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const size_t ELEMENTS_IN_BLOCK = 8;
    const size_t n_column_blocks = (amount_of_columns + ELEMENTS_IN_BLOCK - 1) / ELEMENTS_IN_BLOCK;

    ff_storage<8> state[STATE_SIZE];
    state[0] = ff_dispatch_st<ff_config_starknet>::get_zero();
    state[1] = ff_dispatch_st<ff_config_starknet>::get_zero();
    state[2] = ff_dispatch_st<ff_config_starknet>::get_zero();

    bool has_pending = false;
    ff_storage<8> pending = {0};

    for (size_t block = 0; block < n_column_blocks; block++) {
        // Pack 8 M31 values into a field element
        unsigned __int128 felt[2] = {0, 0};
        size_t start = block * ELEMENTS_IN_BLOCK;
        size_t count = min(ELEMENTS_IN_BLOCK, amount_of_columns - start);

        // Only iterate over actual elements, not padding with zeros
        for (size_t i = 0; i < count; i++) {
            uint32_t limb = columns[start + i][idx] & 0x7FFFFFFF;  // Ensure 31 bits

            unsigned __int128 new_low = (felt[0] << 31) | limb;
            unsigned __int128 new_high = (felt[0] >> (128 - 31)) | (felt[1] << 31);
            felt[0] = new_low;
            felt[1] = new_high;
        }

        // If this is the remainder block, store its length in bits 248, 249 and 250.
        // This matches the CPU implementation in construct_felt252_from_m31s.
        if (count < ELEMENTS_IN_BLOCK) {
            felt[1] += ((unsigned __int128)count) << (248 - 128);
        }

        ff_storage<8> input = {0};
        for (int i = 0; i < 4; i++) {
            input.limbs[i] = (uint32_t)((felt[0] >> (32 * i)) & 0xFFFFFFFF);
        }
        for (int i = 0; i < 4; i++) {
            input.limbs[4 + i] = (uint32_t)((felt[1] >> (32 * i)) & 0xFFFFFFFF);
        }

        if (!has_pending) {
            pending = input;
            has_pending = true;
        } else {
            field_add_device(&state[0], &state[0], &pending);
            field_add_device(&state[1], &state[1], &input);
            poseidon_permute_comp_device(state);
            has_pending = false;
        }
    }

    // Add remainder and domain separation constant
    if (has_pending) {
        field_add_device(&state[0], &state[0], &pending);
    }
    size_t remainder_len = has_pending ? 1 : 0;
    ff_storage<8> one = {0};
    one.limbs[0] = 1;
    field_add_device(&state[remainder_len], &state[remainder_len], &one);
    poseidon_permute_comp_device(state);

    // Convert result to bytes (big-endian)
    uint8_t* output = result + idx * 32;
    for (int i = 0; i < 8; i++) {
        uint32_t limb = state[0].limbs[7 - i];
        output[i * 4 + 0] = (uint8_t)((limb >> 24) & 0xFF);
        output[i * 4 + 1] = (uint8_t)((limb >> 16) & 0xFF);
        output[i * 4 + 2] = (uint8_t)((limb >> 8) & 0xFF);
        output[i * 4 + 3] = (uint8_t)(limb & 0xFF);
    }
}

// CUDA kernel for commit with previous layer
__global__ void poseidon252_commit_on_layer_with_previous_kernel(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    const uint8_t* previous_layer,  // previous layer as bytes
    uint8_t* result  // result as bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const size_t ELEMENTS_IN_BLOCK = 8;
    const size_t n_column_blocks = (amount_of_columns + ELEMENTS_IN_BLOCK - 1) / ELEMENTS_IN_BLOCK;

    ff_storage<8> state[STATE_SIZE];
    state[0] = ff_dispatch_st<ff_config_starknet>::get_zero();
    state[1] = ff_dispatch_st<ff_config_starknet>::get_zero();
    state[2] = ff_dispatch_st<ff_config_starknet>::get_zero();

    bool has_pending = false;
    ff_storage<8> pending = {0};

    // Append left child
    const uint8_t* left_child = previous_layer + (2 * idx) * 32;
    ff_storage<8> left_input = {0};
    for (int i = 0; i < 8; i++) {
        left_input.limbs[i] =
            ((uint32_t)left_child[31 - i * 4] << 0) |
            ((uint32_t)left_child[30 - i * 4] << 8) |
            ((uint32_t)left_child[29 - i * 4] << 16) |
            ((uint32_t)left_child[28 - i * 4] << 24);
    }
    pending = left_input;
    has_pending = true;

    // Append right child and permute
    const uint8_t* right_child = previous_layer + (2 * idx + 1) * 32;
    ff_storage<8> right_input = {0};
    for (int i = 0; i < 8; i++) {
        right_input.limbs[i] =
            ((uint32_t)right_child[31 - i * 4] << 0) |
            ((uint32_t)right_child[30 - i * 4] << 8) |
            ((uint32_t)right_child[29 - i * 4] << 16) |
            ((uint32_t)right_child[28 - i * 4] << 24);
    }
    field_add_device(&state[0], &state[0], &pending);
    field_add_device(&state[1], &state[1], &right_input);
    poseidon_permute_comp_device(state);
    has_pending = false;

    // Process column blocks
    for (size_t block = 0; block < n_column_blocks; block++) {
        unsigned __int128 felt[2] = {0, 0};
        size_t start = block * ELEMENTS_IN_BLOCK;
        size_t count = min(ELEMENTS_IN_BLOCK, amount_of_columns - start);

        // Only iterate over actual elements, not padding with zeros
        for (size_t i = 0; i < count; i++) {
            uint32_t limb = columns[start + i][idx] & 0x7FFFFFFF;  // Ensure 31 bits

            unsigned __int128 new_low = (felt[0] << 31) | limb;
            unsigned __int128 new_high = (felt[0] >> (128 - 31)) | (felt[1] << 31);
            felt[0] = new_low;
            felt[1] = new_high;
        }

        // If this is the remainder block, store its length in bits 248, 249 and 250.
        // This matches the CPU implementation in construct_felt252_from_m31s.
        if (count < ELEMENTS_IN_BLOCK) {
            felt[1] += ((unsigned __int128)count) << (248 - 128);
        }

        ff_storage<8> input = {0};
        for (int i = 0; i < 4; i++) {
            input.limbs[i] = (uint32_t)((felt[0] >> (32 * i)) & 0xFFFFFFFF);
        }
        for (int i = 0; i < 4; i++) {
            input.limbs[4 + i] = (uint32_t)((felt[1] >> (32 * i)) & 0xFFFFFFFF);
        }

        if (!has_pending) {
            pending = input;
            has_pending = true;
        } else {
            field_add_device(&state[0], &state[0], &pending);
            field_add_device(&state[1], &state[1], &input);
            poseidon_permute_comp_device(state);
            has_pending = false;
        }
    }

    // Add remainder and domain separation constant
    if (has_pending) {
        field_add_device(&state[0], &state[0], &pending);
    }
    size_t remainder_len = has_pending ? 1 : 0;
    ff_storage<8> one = {0};
    one.limbs[0] = 1;
    field_add_device(&state[remainder_len], &state[remainder_len], &one);
    poseidon_permute_comp_device(state);

    // Convert result to bytes (big-endian)
    uint8_t* output = result + idx * 32;
    for (int i = 0; i < 8; i++) {
        uint32_t limb = state[0].limbs[7 - i];
        output[i * 4 + 0] = (uint8_t)((limb >> 24) & 0xFF);
        output[i * 4 + 1] = (uint8_t)((limb >> 16) & 0xFF);
        output[i * 4 + 2] = (uint8_t)((limb >> 8) & 0xFF);
        output[i * 4 + 3] = (uint8_t)(limb & 0xFF);
    }
}

// Simple kernel to gather column values for a single position
__global__ void gather_column_values_kernel(
    const uint32_t* const* columns,
    size_t amount_of_columns,
    size_t position,
    uint32_t* output
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < amount_of_columns) {
        output[col] = columns[col][position];
    }
}

// Merkle tree commit functions (GPU-only, using device pointers directly)
extern "C" void poseidon252_commit_on_first_layer(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    Poseidon252Hash* result
) {
    // Initialize constants if not already done
    initialize_poseidon252_constants();

    const unsigned int threads_per_block = 256;
    const unsigned int blocks = (size + threads_per_block - 1) / threads_per_block;

    poseidon252_commit_on_first_layer_kernel<<<blocks, threads_per_block>>>(
        size,
        amount_of_columns,
        columns,
        reinterpret_cast<uint8_t*>(result)
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
}

extern "C" void poseidon252_commit_on_layer_with_previous(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    const Poseidon252Hash* previous_layer,
    Poseidon252Hash* result
) {
    // Initialize constants if not already done
    initialize_poseidon252_constants();

    const unsigned int threads_per_block = 256;
    const unsigned int blocks = (size + threads_per_block - 1) / threads_per_block;

    poseidon252_commit_on_layer_with_previous_kernel<<<blocks, threads_per_block>>>(
        size,
        amount_of_columns,
        columns,
        reinterpret_cast<const uint8_t*>(previous_layer),
        reinterpret_cast<uint8_t*>(result)
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
}
