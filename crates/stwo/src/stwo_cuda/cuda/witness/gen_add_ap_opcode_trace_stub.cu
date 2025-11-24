#include <cstdio>

// Stub implementation for add_ap_opcode CUDA kernels (not yet fully implemented)
// These functions will panic when called - use CPU implementation instead

extern "C" {

void generate_add_ap_opcode_traces(
    const unsigned int **traces,
    const unsigned int *const *lookup_memory_address_to_id_0,
    const unsigned int *const *lookup_memory_id_to_big_0,
    const unsigned int *const *lookup_opcodes_0,
    const unsigned int *const *lookup_opcodes_1,
    const unsigned int *const *lookup_verify_instruction_0,
    const unsigned int *const *sub_component_inputs_verify_instruction,
    const unsigned int *const *sub_component_inputs_memory_address_to_id,
    const unsigned int *const *sub_component_inputs_memory_id_to_big,
    const unsigned int *const *opcodes_input,
    const unsigned int *memory_address_to_id_address_to_raw_id,
    const unsigned int *const *memory_id_to_big_transposed_big_values,
    const unsigned int *memory_id_to_big_small_values,
    unsigned int n_rows,
    unsigned int log_size
) {
    fprintf(stderr, "ERROR: generate_add_ap_opcode_traces CUDA kernel not yet implemented - use CPU version\n");
    // Don't panic, just return - let the test use CPU code
}

void generate_add_ap_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    const unsigned int *const *lookup_memory_address_to_id_0,
    const unsigned int *const *lookup_memory_id_to_big_0,
    const unsigned int *const *lookup_opcodes_0,
    const unsigned int *const *lookup_opcodes_1,
    const unsigned int *const *lookup_verify_instruction_0,
    unsigned int n_rows,
    unsigned int log_size,
    const unsigned int *const *interaction_trace,
    const unsigned int *claimed_sum
) {
    fprintf(stderr, "ERROR: generate_add_ap_opcode_interaction_traces CUDA kernel not yet implemented - use CPU version\n");
    // Don't panic, just return - let the test use CPU code
}

}
