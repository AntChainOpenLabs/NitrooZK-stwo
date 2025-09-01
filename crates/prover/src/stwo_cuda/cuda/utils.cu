#include "utils.cuh"
#include "cuda_mem_pool.cuh"

#include <cstdio>

// Must match the definition in utils.cuh
#define USE_CUDA_MEM_POOL 1


__host__ int log_2(int value) {
    return __builtin_ctz(value);
}

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr, host_ptr, size);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    cuda_mem_copy_host_to_device(host_ptr, device_ptr, size);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst, size);
}

uint32_t* cuda_malloc_uint32_t(int size) {
#if USE_CUDA_MEM_POOL
    uint32_t* device_ptr = cuda_mem_pool_allocate<uint32_t>(size);
    if (device_ptr != nullptr) {
        cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    }
    return device_ptr;
#else
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
#endif
}

Blake2sHash* cuda_malloc_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc<Blake2sHash>(size);
    // cudaMemset(device_ptr, 0x00, sizeof(Blake2sHash) * size);
    return device_ptr;
}

__global__ void print_array(uint32_t *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        printf("%d, ", array[idx]);
    }
}

uint32_t* cuda_alloc_zeroes_uint32_t(int size) {
#if USE_CUDA_MEM_POOL
    return cuda_mem_pool_allocate_zeroes<uint32_t>(size);
#else
    uint32_t* device_ptr = cuda_malloc_uint32_t(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
#endif
}

void cuda_set_uint32_t(uint32_t *device_ptr, size_t index, uint32_t value) {
    cuda_mem_copy_host_to_device<uint32_t>(&value, device_ptr + index, 1);
}

uint32_t cuda_get_uint32_t(uint32_t *device_ptr, size_t index) {
    uint32_t value = 0x0;
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr + index, &value, 1);
    return value;
}

qm31 cuda_get_secure_field(qm31 *device_ptr, size_t index) {
    qm31 value = {};
    cuda_mem_copy_device_to_host<qm31>(device_ptr + index, &value, 1);
    return value;
}

Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc_blake_2s_hash(size);
    cudaMemset(device_ptr, 0x00, sizeof(uint32_t) * size);
    return device_ptr;
}

Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size) {
    Blake2sHash* device_ptr = clone_to_device<Blake2sHash>(host_ptr, size);
    return device_ptr;
}

void cuda_get_blake_2s_hash(Blake2sHash *device_ptr, Blake2sHash *host_ptr, size_t index) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr + index, host_ptr, 1);
}

void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr, host_ptr, size);
}

void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size) {
    cuda_mem_copy_device_to_device<Blake2sHash>(from, dst, size);
}

uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size) {
    uint32_t** device_ptr = clone_to_device<uint32_t*>(host_ptr, size);
    return device_ptr;
}

// void** copy_device_pointer_vec_from_host_to_device(const void** ptrs, size_t n) {
//     void** d_ptrs;
//     cudaMalloc(&d_ptrs, n * sizeof(void*));
//     cudaMemcpy(d_ptrs, ptrs, n * sizeof(void*), cudaMemcpyHostToDevice);
//     return d_ptrs;
// }

void cuda_free_memory(void *device_ptr) {
#if USE_CUDA_MEM_POOL
    cuda_mem_pool_free(device_ptr);
#else
    cudaError_t err = cudaFree(device_ptr);
    if (err != cudaSuccess) {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
#endif
}

// Stub implementations for backward compatibility
// These will be removed once all code is migrated to use CUDA memory pool directly
extern "C" uint32_t* pool_allocate_cuda(size_t size) {
    return cuda_mem_pool_allocate_uint32(size);
}

extern "C" void pool_deallocate_cuda(uint32_t* ptr, size_t size) {
    (void)size; // Unused parameter
    cuda_mem_pool_free_uint32(ptr);
}

extern "C" uint32_t* pool_allocate_zeroes_cuda(size_t size) {
    return cuda_mem_pool_allocate_zeroes_uint32(size);
}
