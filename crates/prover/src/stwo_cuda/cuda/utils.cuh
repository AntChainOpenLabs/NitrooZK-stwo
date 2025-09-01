#ifndef UTILS_H
#define UTILS_H

#include "fields.cuh"
#include <cstdio>
#include <unordered_map>

#include <cuda_runtime.h>

#include <cstdint>

#ifdef __CUDA_ARCH__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#define DEVICE_FORCEINLINE __device__ __forceinline__

#define HOST_DEVICE_FORCEINLINE __host__ __device__ __forceinline__

#define EXTERN extern "C" [[maybe_unused]]

#ifndef ASSERT_CUDA_SUCCESS
static void handle_cuda_error(cudaError_t cuda_error, const char *const file,
                              int const line) {
  if (cuda_error != cudaError::cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d error=%s message: %s \n", file, line,
            cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));
    exit(1);
  }
}
#define ASSERT_CUDA_SUCCESS(error) handle_cuda_error(error, __FILE__, __LINE__)
#endif

#define HANDLE_CUDA_ERROR(statement)                                                                                                                           \
  {                                                                                                                                                            \
    cudaError_t hce_result = (statement);                                                                                                                      \
    if (hce_result != cudaSuccess)                                                                                                                             \
      printf("line : %d, file : %s, error_code:%d, error: %s \n", __LINE__, __FILE__, hce_result, cudaGetErrorString(hce_result));                                                          \
    if (hce_result != cudaSuccess)                                                                                                                             \
      return hce_result;                                                                                                                                       \
  }

#ifndef ASSERT_TRUE
HOST_DEVICE_FORCEINLINE void assert_true(bool condition, const char *message,
                                         const char *const file,
                                         int const line) {
#ifdef __CUDA_ARCH__
  if (condition == false) {
    printf("Error at %s:%d: %s, tid = %u\n", file, line, message,
           threadIdx.x + blockIdx.x * blockDim.x);
  }
#else
  if (condition == false) {
    printf("Error at %s:%d: %s\n", file, line, message);
    exit(1);
  }
#endif
}
#define ASSERT_TRUE(condition, msg) \
  assert_true(condition, msg, __FILE__, __LINE__)
#endif

struct Blake2sHash {
    unsigned int s[8];
};

DEVICE_FORCEINLINE uint32_t bit_reverse(uint32_t n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

DEVICE_FORCEINLINE unsigned int offset_bit_reversed_circle_domain_index(
    unsigned int i,
    unsigned int domain_log_size,
    unsigned int eval_log_size,
    int offset
) {
    unsigned int prev_index = bit_reverse(i, eval_log_size);
    unsigned int half_size = 1 << (eval_log_size - 1);
    int step_size = offset * (1 << (eval_log_size - domain_log_size - 1));

    if (prev_index < half_size) {
        prev_index = (prev_index + step_size) % half_size;
    } else {
        prev_index = ((prev_index - step_size) % half_size) + half_size;
    }

    return bit_reverse(prev_index, eval_log_size);
}


__host__ int log_2(int value);

extern "C"
void copy_uint32_t_vec_from_device_to_host(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t*, int);

extern "C"
void copy_uint32_t_vec_from_device_to_device(uint32_t *, uint32_t*, int);

extern "C"
uint32_t* cuda_malloc_uint32_t(int);

extern "C"
Blake2sHash* cuda_malloc_blake_2s_hash(int);

extern "C"
uint32_t* cuda_alloc_zeroes_uint32_t(int);

#include "cuda_mem_pool.cuh"

extern "C"
void cuda_set_uint32_t(uint32_t *device_ptr, size_t index, uint32_t value);

extern "C"
uint32_t cuda_get_uint32_t(uint32_t *device_ptr, size_t index);

extern "C"
qm31 cuda_get_secure_field(uint32_t *device_ptr, size_t index);

extern "C"
Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int);

extern "C"
void cuda_free_memory(void*);

// Use CUDA memory pool
#define USE_CUDA_MEM_POOL 1

template<typename T>
T* cuda_malloc(unsigned int size) {
#if USE_CUDA_MEM_POOL
    return cuda_mem_pool_allocate<T>(size);
#else
    T *device_ptr;
    cudaError_t err = cudaMalloc((void**)&device_ptr, sizeof(T) * size);
    if (err != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
    }
    return device_ptr;
#endif
}

template<typename T>
void cuda_mem_copy_host_to_device(T* host_data, T* device_data, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(device_data, host_data, sizeof(T) * data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void cuda_mem_copy_device_to_device(T* device_data_from, T* device_data_to, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(device_data_to, device_data_from, sizeof(T) * data_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void cuda_mem_copy_device_to_host(T* device_data, T* host_data, unsigned int data_size) {
    cudaError_t err = cudaMemcpy(host_data, device_data, sizeof(T) * data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
T* clone_to_device(T* host_data, unsigned int data_size) {
    T* device_data = cuda_malloc<T>(data_size);
    cuda_mem_copy_host_to_device(host_data, device_data, data_size);
    return device_data;
}


extern "C"
Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size);

extern "C"
void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size);

extern "C"
void cuda_get_blake_2s_hash(Blake2sHash *device_ptr, Blake2sHash *host_ptr, size_t index);

extern "C"
uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size);

#define THREAD_COUNT_MAX 1024

#endif // UTILS_H
