// Elliptic curve operations for Starknet curve
// Used for Pedersen hash computation
//
// Curve: y^2 = x^3 + alpha*x + beta over Starknet field
// alpha = 1, beta = 0x6f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89

#ifndef EC_OPS_CUH
#define EC_OPS_CUH

#include "fp256_config.cuh"
#include "fp256_dispatch_st.cuh"
#include "fields.cuh"

// Type alias for 256-bit field element (Felt252)
typedef ff_storage<8> felt252;

// Affine point on the Starknet curve
struct AffinePointCuda {
    felt252 x;
    felt252 y;
};

// Projective point on the Starknet curve (X, Y, Z) where x = X/Z, y = Y/Z
struct ProjectivePointCuda {
    felt252 X;
    felt252 Y;
    felt252 Z;
};

// Field operations using Starknet config
__device__ __forceinline__ felt252 felt_add(const felt252& a, const felt252& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

__device__ __forceinline__ felt252 felt_sub(const felt252& a, const felt252& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

__device__ __forceinline__ felt252 felt_mul(const felt252& a, const felt252& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

__device__ __forceinline__ felt252 felt_to_mont(const felt252& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

__device__ __forceinline__ felt252 felt_from_mont(const felt252& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

__device__ __forceinline__ felt252 felt_inverse(const felt252& a) {
    return ff_dispatch_st<ff_config_starknet>::inverse(a);
}

// Check if projective point is at infinity (Z = 0)
__device__ __forceinline__ bool is_infinity(const ProjectivePointCuda& p) {
    // Check if Z is zero
    for (int i = 0; i < 8; i++) {
        if (p.Z.limbs[i] != 0) return false;
    }
    return true;
}

// Create projective point from affine point
__device__ __forceinline__ ProjectivePointCuda affine_to_projective(const AffinePointCuda& p) {
    ProjectivePointCuda result;
    result.X = p.x;
    result.Y = p.y;
    // Z = 1 in Montgomery form
    result.Z = ff_config_starknet::one;
    return result;
}

// Mixed addition: Projective + Affine -> Projective
// Uses the formula for adding an affine point to a projective point
// This is more efficient than general projective addition
//
// Input: P = (X1, Y1, Z1) in projective, Q = (x2, y2) in affine
// Output: P + Q in projective
//
// Algorithm (from https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo):
// u = Y2*Z1-Y1
// uu = u^2
// v = X2*Z1-X1
// vv = v^2
// vvv = v*vv
// R = vv*X1
// A = uu*Z1-vvv-2*R
// X3 = v*A
// Y3 = u*(R-A)-vvv*Y1
// Z3 = vvv*Z1
__device__ void ec_add_mixed(
    ProjectivePointCuda& P,  // Input/Output projective point
    const AffinePointCuda& Q  // Input affine point (from table)
) {
    // All operations in Montgomery form
    felt252 X1 = P.X;
    felt252 Y1 = P.Y;
    felt252 Z1 = P.Z;

    // Convert Q to Montgomery form
    felt252 X2 = felt_to_mont(Q.x);
    felt252 Y2 = felt_to_mont(Q.y);

    // u = Y2*Z1 - Y1
    felt252 u = felt_sub(felt_mul(Y2, Z1), Y1);

    // v = X2*Z1 - X1
    felt252 v = felt_sub(felt_mul(X2, Z1), X1);

    // uu = u^2
    felt252 uu = felt_mul(u, u);

    // vv = v^2
    felt252 vv = felt_mul(v, v);

    // vvv = v * vv
    felt252 vvv = felt_mul(v, vv);

    // R = vv * X1
    felt252 R = felt_mul(vv, X1);

    // A = uu*Z1 - vvv - 2*R
    felt252 uu_Z1 = felt_mul(uu, Z1);
    felt252 two_R = felt_add(R, R);
    felt252 A = felt_sub(felt_sub(uu_Z1, vvv), two_R);

    // X3 = v * A
    P.X = felt_mul(v, A);

    // Y3 = u*(R-A) - vvv*Y1
    felt252 R_minus_A = felt_sub(R, A);
    felt252 u_R_A = felt_mul(u, R_minus_A);
    felt252 vvv_Y1 = felt_mul(vvv, Y1);
    P.Y = felt_sub(u_R_A, vvv_Y1);

    // Z3 = vvv * Z1
    P.Z = felt_mul(vvv, Z1);
}

// Convert projective point to affine (requires field inversion)
// Computes x = X/Z, y = Y/Z using field inversion
// This is expensive (~251 squarings + ~128 multiplications), so should be done sparingly
__device__ void projective_to_affine(
    const ProjectivePointCuda& P,
    AffinePointCuda& result
) {
    // Compute Z^(-1) using extended Euclidean algorithm
    felt252 z_inv = felt_inverse(P.Z);

    // result.x = X * Z^(-1) (in Montgomery form)
    // result.y = Y * Z^(-1) (in Montgomery form)
    felt252 x_mont = felt_mul(P.X, z_inv);
    felt252 y_mont = felt_mul(P.Y, z_inv);

    // Convert from Montgomery form to standard form
    result.x = felt_from_mont(x_mont);
    result.y = felt_from_mont(y_mont);
}

// Reconstruct Felt252 from 28 M31 limbs (9-bit each)
// The Felt252 is stored as 28 * 9 = 252 bits split into 28 M31 values
__device__ void felt252_from_m31_limbs(felt252& result, const m31* limbs) {
    // Clear result
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = 0;
    }

    // Each M31 limb is 9 bits
    // We need to combine them into 8 * 32 = 256 bits
    // Limb layout: limbs[0] = bits 0-8, limbs[1] = bits 9-17, etc.

    uint64_t accumulator = 0;
    int bit_pos = 0;
    int result_limb = 0;

    for (int i = 0; i < 28 && result_limb < 8; i++) {
        uint64_t value = limbs[i];
        accumulator |= (value << (bit_pos % 32));
        bit_pos += 9;

        // Extract 32-bit limbs when we have enough bits
        while (bit_pos >= 32 && result_limb < 8) {
            result.limbs[result_limb] = (uint32_t)(accumulator & 0xFFFFFFFF);
            accumulator >>= 32;
            bit_pos -= 32;
            result_limb++;
        }
    }

    // Handle remaining bits
    if (result_limb < 8 && accumulator != 0) {
        result.limbs[result_limb] = (uint32_t)accumulator;
    }
}

// Store Felt252 into 28 M31 limbs (9-bit each)
__device__ void felt252_to_m31_limbs(const felt252& value, m31* limbs) {
    // Extract 9 bits at a time from the 256-bit value
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int limb_idx = 0;
    int result_idx = 0;

    while (result_idx < 28) {
        // Load more bits if needed
        while (bits_in_acc < 9 && limb_idx < 8) {
            accumulator |= ((uint64_t)value.limbs[limb_idx]) << bits_in_acc;
            bits_in_acc += 32;
            limb_idx++;
        }

        // Extract 9 bits
        limbs[result_idx] = (m31){(uint32_t)(accumulator & 0x1FF)};
        accumulator >>= 9;
        bits_in_acc -= 9;
        result_idx++;
    }
}

#endif // EC_OPS_CUH
