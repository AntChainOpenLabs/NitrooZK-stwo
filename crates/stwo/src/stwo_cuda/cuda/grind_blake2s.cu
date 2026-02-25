#include "grind_blake2s.cuh"
#include "utils.cuh"

// Blake2s IV constants (same as in blake2s.cu, but we need our own copy
// since __device__ __constant__ variables have translation-unit scope)
static __device__ __constant__ uint32_t grind_blake2s_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static __device__ __constant__ uint8_t grind_blake2s_sigma[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};

#define GRIND_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define GRIND_G(r,i,a,b,c,d) \
    do { \
        a = a + b + m[grind_blake2s_sigma[r][2*i+0]]; \
        d = GRIND_ROTR32(d ^ a, 16); \
        c = c + d; \
        b = GRIND_ROTR32(b ^ c, 12); \
        a = a + b + m[grind_blake2s_sigma[r][2*i+1]]; \
        d = GRIND_ROTR32(d ^ a, 8); \
        c = c + d; \
        b = GRIND_ROTR32(b ^ c, 7); \
    } while(0)

// Each thread tests one nonce. The kernel performs a single-block Blake2s hash:
//   H(prefixed_digest || nonce)
// where prefixed_digest is 32 bytes (8 x u32) and nonce is 8 bytes (u64),
// giving a 40-byte single-block message (fits in one 64-byte Blake2s block).
//
// If hash[0] has >= pow_bits trailing zeros, the nonce is valid.
// We use atomicMin to track the smallest valid nonce across all threads.
__global__ void grind_blake2s_kernel(
    const uint32_t* __restrict__ prefixed_digest,
    uint32_t pow_bits,
    uint64_t nonce_offset,
    unsigned long long* result_nonce
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = nonce_offset + tid;

    // Early exit: if a smaller nonce was already found, skip this thread
    if (nonce >= *result_nonce) return;

    // Build 16-word message block:
    // m[0..7]  = prefixed_digest (32 bytes)
    // m[8]     = nonce low 32 bits
    // m[9]     = nonce high 32 bits
    // m[10..15] = 0 (padding)
    // Total input length = 40 bytes
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = prefixed_digest[i];
    m[8]  = (uint32_t)(nonce);
    m[9]  = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; i++) m[i] = 0;

    // Initialize Blake2s state for a single-block hash (32-byte output):
    //   h[0] = IV[0] ^ 0x01010020 (fan-out=1, depth=1, digest_length=32)
    //   h[1..7] = IV[1..7]
    // Initialize v-vector for compression:
    //   v[0..7] = h[0..7]
    //   v[8..15] = IV[0..7]
    //   v[12] ^= 40     (total bytes counter, low word)
    //   v[14] ^= 0xFFFFFFFF  (last block flag)
    uint32_t v[16];
    v[0] = grind_blake2s_IV[0] ^ 0x01010020;
    #pragma unroll
    for (int i = 1; i < 8; i++) v[i] = grind_blake2s_IV[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i + 8] = grind_blake2s_IV[i];
    v[12] ^= 40;          // t0 = 40 bytes total
    v[14] ^= 0xFFFFFFFF;  // last block

    // 10 rounds of Blake2s mixing
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        GRIND_G(r, 0, v[0], v[4], v[8],  v[12]);
        GRIND_G(r, 1, v[1], v[5], v[9],  v[13]);
        GRIND_G(r, 2, v[2], v[6], v[10], v[14]);
        GRIND_G(r, 3, v[3], v[7], v[11], v[15]);
        GRIND_G(r, 4, v[0], v[5], v[10], v[15]);
        GRIND_G(r, 5, v[1], v[6], v[11], v[12]);
        GRIND_G(r, 6, v[2], v[7], v[8],  v[13]);
        GRIND_G(r, 7, v[3], v[4], v[9],  v[14]);
    }

    // Finalize: h[i] = h[i] ^ v[i] ^ v[i+8], but we only need h[0]
    uint32_t h0 = (grind_blake2s_IV[0] ^ 0x01010020) ^ v[0] ^ v[8];

    // Count trailing zeros of h0
    // __clz counts leading zeros, __brev reverses bits
    // For h0 == 0, __clz returns 32, which means 32 trailing zeros
    uint32_t tz = (h0 == 0) ? 32 : __clz(__brev(h0));

    if (tz >= pow_bits) {
        atomicMin(result_nonce, (unsigned long long)nonce);
    }
}

// Host function: launches the GPU grind kernel in batches until a valid nonce is found.
// Returns the smallest valid nonce.
uint64_t grind_blake2s(const uint32_t* host_prefixed_digest, uint32_t pow_bits) {
    // Allocate device memory for prefixed_digest (8 x u32)
    uint32_t* d_prefixed_digest;
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_prefixed_digest, 8 * sizeof(uint32_t)));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_prefixed_digest, host_prefixed_digest,
                                   8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate device memory for result nonce, initialize to UINT64_MAX
    unsigned long long* d_result;
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_result, sizeof(unsigned long long)));
    unsigned long long init_val = UINT64_MAX;
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_result, &init_val,
                                   sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int block_size = 256;
    const int grid_size = 4096;  // 4096 blocks * 256 threads = 1,048,576 nonces per batch
    const uint64_t batch_size = (uint64_t)block_size * grid_size;

    uint64_t nonce_offset = 0;
    unsigned long long host_result = UINT64_MAX;

    while (host_result == UINT64_MAX) {
        grind_blake2s_kernel<<<grid_size, block_size>>>(
            d_prefixed_digest, pow_bits, nonce_offset, d_result
        );
        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // Check if a valid nonce was found
        ASSERT_CUDA_SUCCESS(cudaMemcpy(&host_result, d_result,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        nonce_offset += batch_size;
    }

    ASSERT_CUDA_SUCCESS(cudaFree(d_prefixed_digest));
    ASSERT_CUDA_SUCCESS(cudaFree(d_result));

    return (uint64_t)host_result;
}
