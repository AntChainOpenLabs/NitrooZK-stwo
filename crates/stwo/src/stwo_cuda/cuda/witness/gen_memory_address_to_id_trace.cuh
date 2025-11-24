
#ifndef GEN_MEMORY_ADDRESS_TO_ID_TRACE_H
#define GEN_MEMORY_ADDRESS_TO_ID_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "gen_blake_round_sigma_trace.cuh"

struct address_to_raw_id {
    unsigned *device_ptr;
    size_t size;
};

struct multiplicities {
    unsigned *device_ptr;
    size_t size;
};

DEVICE_FORCEINLINE void memory_address_to_id_deduce_output(
    unsigned* address_to_raw_id,
    m31 input,
    m31 *output
) {
    unsigned indices = input;
    *output = (m31) address_to_raw_id[indices - 1];
}

extern "C"
void memory_address_to_id_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

#endif // GEN_MEMORY_ADDRESS_TO_ID_TRACE_H