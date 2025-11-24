
#ifndef GEN_MEMORY_ID_TO_ADDRESS_TRACE_H
#define GEN_MEMORY_ID_TO_ADDRESS_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "gen_blake_round_sigma_trace.cuh"
#include "gen_memory.cuh"

#define N_M31_IN_FELT252 28
#define N_M31_IN_SMALL_FELT252  8
#define N_BITS_PER_FELT  9

struct Felt252 {
    m31 value[N_M31_IN_FELT252];
};

struct memory_id_to_big_base_vector {
    unsigned *device_ptr;
    size_t size;
};

struct memory_id_to_big_claim_generator {
    struct memory_id_to_big_base_vector transposed_big_values[8];
    struct memory_id_to_big_base_vector big_mults;
    struct memory_id_to_big_base_vector small_values;
    struct memory_id_to_big_base_vector small_mults;
};


HOST_DEVICE_FORCEINLINE void split_f252(const uint32_t x[8], m31 out[N_M31_IN_FELT252]) {
    const uint32_t mask = (1 << N_BITS_PER_FELT) - 1;
    uint32_t word_idx = 0;
    uint32_t bits_remaining = 32;
    uint32_t current_word = x[0];

    for (int i = 0; i < N_M31_IN_FELT252; ++i) {
        uint32_t value = 0;
        uint32_t bits_needed = N_BITS_PER_FELT;

        if (bits_remaining >= bits_needed) {
            value = current_word & mask;
            current_word >>= N_BITS_PER_FELT;
            bits_remaining -= N_BITS_PER_FELT;
        } else {
            value = current_word;
            bits_needed -= bits_remaining;

            if (++word_idx >= 8) {
                current_word = 0;
            } else {
                current_word = x[word_idx];
            }

            value |= (current_word & ((1 << bits_needed) - 1)) << bits_remaining;
            current_word >>= bits_needed;
            bits_remaining = 32 - bits_needed;
        }

        out[i] = m31{value};
    }
}

HOST_DEVICE_FORCEINLINE void memory_id_to_big_state_deduce_output(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
) {
    unsigned out_values[8];
    EncodedMemoryValueId emv;
    emv.encoded = id;

    MemoryValueId mv = decode_memory_value_id(&emv);

    for (int j = 0; j < 8; ++j) {
        switch (mv.tag) {
            case MEMORY_VALUE_ID_F252: {
                out_values[j] = transpose_big_value_ptr[j][mv.value];
                break;
            }
            case MEMORY_VALUE_ID_SMALL: {
                if (j >= 4) {
                    out_values[j] = 0;
                } else {
                    uint32_t limbs[4];
                    u128_to_4_limbs(((u128 *)small_value_ptr)[mv.value], limbs);
                    out_values[j] = limbs[j];
                }
                break;
            }
            case MEMORY_VALUE_ID_EMPTY: {
                printf("Attempted deduce_output on empty memory cell.\\n");
                return;
            }
            default: {
                printf("Invalid MemoryValueId tag: %d\\n", mv.tag);
                return;
            }
        }
    }

    split_f252(out_values, felt252_out);
}

extern "C"
void memory_id_to_big_deduce_finese_cuda(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
);

extern "C"
void memory_id_to_big_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *big_mults,
    unsigned big_mults_row_log_size,
    m31 *small_mults,
    unsigned small_mults_row_log_size
);
#endif // GEN_MEMORY_ID_TO_ADDRESS_TRACE_H