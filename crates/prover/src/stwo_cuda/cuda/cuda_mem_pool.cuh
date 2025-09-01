#ifndef CUDA_MEM_POOL_H
#define CUDA_MEM_POOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Global memory pool handle
extern cudaMemPool_t g_mem_pool;
extern bool g_mem_pool_initialized;

// Initialize the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_init();

// Destroy the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_destroy();

// Allocate memory from the pool
template<typename T>
T* cuda_mem_pool_allocate(size_t count) {
    T* ptr = nullptr;
    size_t size = sizeof(T) * count;
    
    if (!g_mem_pool_initialized) {
        cudaError_t err = cuda_mem_pool_init();
        if (err != cudaSuccess) {
            printf("Failed to initialize memory pool: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
    }
    
    cudaError_t err = cudaMallocFromPoolAsync((void**)&ptr, size, g_mem_pool, 0);
    if (err != cudaSuccess) {
        printf("Failed to allocate %zu bytes from pool: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    
    // Synchronize to ensure allocation is complete
    cudaStreamSynchronize(0);
    
    return ptr;
}

// Allocate zeroed memory from the pool
template<typename T>
T* cuda_mem_pool_allocate_zeroes(size_t count) {
    T* ptr = cuda_mem_pool_allocate<T>(count);
    if (ptr != nullptr) {
        cudaMemsetAsync(ptr, 0, sizeof(T) * count, 0);
        cudaStreamSynchronize(0);
    }
    return ptr;
}

// Free memory back to the pool
template<typename T>
void cuda_mem_pool_free(T* ptr) {
    if (ptr != nullptr) {
        cudaFreeAsync(ptr, 0);
        cudaStreamSynchronize(0);
    }
}

// C-style wrappers for specific types
extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count);
extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count);
extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr);

#endif // CUDA_MEM_POOL_H