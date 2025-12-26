use std::ffi::c_void;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};

#[link(name = "stwo_cuda")]
extern "C" {

    pub fn verify_bitwise_xor_4_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
     );

     pub fn verify_bitwise_xor_7_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
     );

    pub fn verify_bitwise_xor_8_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
     );

    pub fn verify_bitwise_xor_8_b_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
     );

     pub fn verify_bitwise_xor_9_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
     );

    pub fn verify_bitwise_xor_12_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const *const u32,
        mults_col_size: u32,
        mults_row_log_size: u32,
    );

    pub fn generate_blake_g_traces(
        traces: *const *const u32,
        lookup_blake_g_0: *const *const u32,
        lookup_verify_bitwise_xor_12_0: *const *const u32,
        lookup_verify_bitwise_xor_12_1: *const *const u32,
        lookup_verify_bitwise_xor_4_0 : *const *const u32,
        lookup_verify_bitwise_xor_4_1 : *const *const u32,
        lookup_verify_bitwise_xor_7_0 : *const *const u32,
        lookup_verify_bitwise_xor_7_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_3 : *const *const u32,
        lookup_verify_bitwise_xor_8_4 : *const *const u32,
        lookup_verify_bitwise_xor_8_5 : *const *const u32,
        lookup_verify_bitwise_xor_8_6 : *const *const u32,
        lookup_verify_bitwise_xor_8_7 : *const *const u32,
        lookup_verify_bitwise_xor_9_0 : *const *const u32,
        lookup_verify_bitwise_xor_9_1 : *const *const u32,

        sub_componet_input_verify_bitwise_xor_8: *const *const u32,
        sub_componet_input_verify_bitwise_xor_12: *const *const u32,
        sub_componet_input_verify_bitwise_xor_4: *const *const u32,
        sub_componet_input_verify_bitwise_xor_7: *const *const u32,
        sub_componet_input_verify_bitwise_xor_9: *const *const u32,

        blake_g_input : *const *const u32,

        trace_log_len: u32,
    );

    pub fn generate_blake_g_interaction_traces(
        blake_g: *mut c_void,
        verify_bitwise_xor_12: *mut c_void,
        verify_bitwise_xor_4 : *mut c_void,
        verify_bitwise_xor_7 : *mut c_void,
        verify_bitwise_xor_8 : *mut c_void,
        verify_bitwise_xor_9 : *mut c_void,

        lookup_blake_g_0              : *const *const u32,
        lookup_verify_bitwise_xor_12_0: *const *const u32,
        lookup_verify_bitwise_xor_12_1: *const *const u32,
        lookup_verify_bitwise_xor_4_0 : *const *const u32,
        lookup_verify_bitwise_xor_4_1 : *const *const u32,
        lookup_verify_bitwise_xor_7_0 : *const *const u32,
        lookup_verify_bitwise_xor_7_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_3 : *const *const u32,
        lookup_verify_bitwise_xor_8_4 : *const *const u32,
        lookup_verify_bitwise_xor_8_5 : *const *const u32,
        lookup_verify_bitwise_xor_8_6 : *const *const u32,
        lookup_verify_bitwise_xor_8_7 : *const *const u32,
        lookup_verify_bitwise_xor_9_0 : *const *const u32,
        lookup_verify_bitwise_xor_9_1 : *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn generate_triple_xor_32_traces(
        traces: *const *const u32,
        lookup_triple_xor_32: *const *const u32,
        lookup_verify_bitwise_xor_8_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_3 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_3 : *const *const u32,

        sub_componet_input_verify_bitwise_xor_8: *const *const u32,
        sub_componet_input_verify_bitwise_xor_8_b: *const *const u32,

        triple_xor_32_input : *const *const u32,

        trace_log_len: u32,
    );

    pub fn generate_triple_xor_32_interaction_traces(
        triple_xor_32: *mut c_void,
        verify_bitwise_xor_8 : *mut c_void,
        verify_bitwise_xor_8_b : *mut c_void,

        lookup_triple_xor_32          : *const *const u32,
        lookup_verify_bitwise_xor_8_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_3 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_0 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_1 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_2 : *const *const u32,
        lookup_verify_bitwise_xor_8_b_3 : *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Blake round trace generation
    pub fn generate_blake_round_traces(
        traces: *const *const u32,

        lookup_blake_g_0: *const *const u32,
        lookup_blake_g_1: *const *const u32,
        lookup_blake_g_2: *const *const u32,
        lookup_blake_g_3: *const *const u32,
        lookup_blake_g_4: *const *const u32,
        lookup_blake_g_5: *const *const u32,
        lookup_blake_g_6: *const *const u32,
        lookup_blake_g_7: *const *const u32,
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,
        lookup_blake_round_sigma_0: *const *const u32,
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,

        sub_component_inputs_blake_round_sigma: *const *const u32,
        sub_component_inputs_range_check_7_2_5: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_blake_g: *const *const u32,

        blake_round_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transpose_big_value_ptr: *const *const u32,
        memory_id_to_big_small_value_ptr: *const u32,

        trace_log_len: u32,
    );

    pub fn generate_blake_round_interaction_traces(
        blake_g: *mut c_void,
        blake_round: *mut c_void,
        blake_round_sigma: *mut c_void,
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_7_2_5: *mut c_void,

        lookup_blake_g_0: *const *const u32,
        lookup_blake_g_1: *const *const u32,
        lookup_blake_g_2: *const *const u32,
        lookup_blake_g_3: *const *const u32,
        lookup_blake_g_4: *const *const u32,
        lookup_blake_g_5: *const *const u32,
        lookup_blake_g_6: *const *const u32,
        lookup_blake_g_7: *const *const u32,

        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        lookup_blake_round_sigma_0: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,

        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,

        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn blake_round_sigma_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const *const u32,
        mults_cols_sizes: u32,
        mults_row_log_size: u32,
     );

     pub fn generate_blake_round_sigma_interaction_traces(
        blake_round_sigma: *mut c_void,

        lookup_blake_round_sigma : *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Memory address to ID functions
    pub fn memory_address_to_id_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn generate_memory_address_to_id_traces(
        traces: *const *const u32,
        interaction_traces: *const *const u32,
        address_to_raw_id: *const u32,
        multiplicities: *const u32,
        total_size_log: u32,
        log_size: u32,
        lookup_elements: *mut c_void,
    );

    // Memory ID to big functions
    pub fn memory_id_to_big_deduce_finese_cuda(
        transpose_big_value_ptr: *const *const u32,
        small_value_ptr: *const u32,
        id: u32,
        felt252_out: *const u32,
    );

    pub fn memory_id_to_big_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        big_mults: *const u32,
        big_mults_row_log_size: u32,
        small_mults: *const u32,
        small_mults_row_log_size: u32,
    );

    // Add opcode trace generation
    pub fn generate_add_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // add_opcode_small functions
    pub fn generate_add_opcode_small_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_opcode_small_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode functions
    pub fn generate_assert_eq_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode_double_deref functions
    pub fn generate_assert_eq_opcode_double_deref_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,

        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_double_deref_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,

        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode_imm functions
    pub fn generate_assert_eq_opcode_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // add_ap_opcode functions
    pub fn generate_add_ap_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_ap_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_11: *mut c_void,
        range_check_18: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // Range check vector functions
    pub fn range_check_vector_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        n_range: u32,
        ranges: *const u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn partition_into_bit_segments_cuda(
        total_size: u32,
        n_range: u32,
        n_bits_per_segments: *const u32,
        output_value: *const *const u32,
    );

    pub fn range_check_vector_generate_interaction_trace(
        cols_tmp_vec: *const *const u32,
        n_range: u32,
        ranges: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    // blake_compress_opcode functions
    pub fn generate_blake_compress_opcode_traces(
        traces: *const *const u32,

        // Lookup data - blake_round (2 lookups)
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        // Lookup data - memory_address_to_id (20 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,

        // Lookup data - memory_id_to_big (20 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_7_2_5 (17 lookups)
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,
        lookup_range_check_7_2_5_16: *const *const u32,

        // Lookup data - triple_xor_32 (8 lookups)
        lookup_triple_xor_32_0: *const *const u32,
        lookup_triple_xor_32_1: *const *const u32,
        lookup_triple_xor_32_2: *const *const u32,
        lookup_triple_xor_32_3: *const *const u32,
        lookup_triple_xor_32_4: *const *const u32,
        lookup_triple_xor_32_5: *const *const u32,
        lookup_triple_xor_32_6: *const *const u32,
        lookup_triple_xor_32_7: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (4 lookups)
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_7_2_5: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_8: *const *const u32,
        sub_component_inputs_blake_round: *const *const u32,
        sub_component_inputs_triple_xor_32: *const *const u32,

        // Opcode inputs
        blake_compress_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_blake_compress_opcode_interaction_traces(
        // Relations
        blake_round: *mut c_void,
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_7_2_5: *mut c_void,
        triple_xor_32: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,
        verify_instruction: *mut c_void,

        // Lookup data - blake_round (2 lookups)
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        // Lookup data - memory_address_to_id (20 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,

        // Lookup data - memory_id_to_big (20 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_7_2_5 (17 lookups)
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,
        lookup_range_check_7_2_5_16: *const *const u32,

        // Lookup data - triple_xor_32 (8 lookups)
        lookup_triple_xor_32_0: *const *const u32,
        lookup_triple_xor_32_1: *const *const u32,
        lookup_triple_xor_32_2: *const *const u32,
        lookup_triple_xor_32_3: *const *const u32,
        lookup_triple_xor_32_4: *const *const u32,
        lookup_triple_xor_32_5: *const *const u32,
        lookup_triple_xor_32_6: *const *const u32,
        lookup_triple_xor_32_7: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (4 lookups)
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // call_opcode functions
    pub fn generate_call_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        call_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_call_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // call_opcode_rel_imm functions
    pub fn generate_call_opcode_rel_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        call_opcode_rel_imm_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_call_opcode_rel_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // generic_opcode functions (244 trace columns, 34 interaction columns, 67 lookups)
    pub fn generate_generic_opcode_traces(
        traces: *const *const u32,

        // memory_address_to_id (3 lookups × 2 fields)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // memory_id_to_big (3 lookups × 29 fields)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // opcodes (2 lookups × 3 fields)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // range_check_9_9 (4 lookups × 2 fields)
        lookup_range_check_9_9_0: *const *const u32,
        lookup_range_check_9_9_1: *const *const u32,
        lookup_range_check_9_9_2: *const *const u32,
        lookup_range_check_9_9_3: *const *const u32,

        // range_check_9_9_b (4 lookups × 2 fields)
        lookup_range_check_9_9_b_0: *const *const u32,
        lookup_range_check_9_9_b_1: *const *const u32,
        lookup_range_check_9_9_b_2: *const *const u32,
        lookup_range_check_9_9_b_3: *const *const u32,

        // range_check_9_9_c (4 lookups × 2 fields)
        lookup_range_check_9_9_c_0: *const *const u32,
        lookup_range_check_9_9_c_1: *const *const u32,
        lookup_range_check_9_9_c_2: *const *const u32,
        lookup_range_check_9_9_c_3: *const *const u32,

        // range_check_9_9_d (4 lookups × 2 fields)
        lookup_range_check_9_9_d_0: *const *const u32,
        lookup_range_check_9_9_d_1: *const *const u32,
        lookup_range_check_9_9_d_2: *const *const u32,
        lookup_range_check_9_9_d_3: *const *const u32,

        // range_check_9_9_e (4 lookups × 2 fields)
        lookup_range_check_9_9_e_0: *const *const u32,
        lookup_range_check_9_9_e_1: *const *const u32,
        lookup_range_check_9_9_e_2: *const *const u32,
        lookup_range_check_9_9_e_3: *const *const u32,

        // range_check_9_9_f (4 lookups × 2 fields)
        lookup_range_check_9_9_f_0: *const *const u32,
        lookup_range_check_9_9_f_1: *const *const u32,
        lookup_range_check_9_9_f_2: *const *const u32,
        lookup_range_check_9_9_f_3: *const *const u32,

        // range_check_9_9_g (2 lookups × 2 fields)
        lookup_range_check_9_9_g_0: *const *const u32,
        lookup_range_check_9_9_g_1: *const *const u32,

        // range_check_9_9_h (2 lookups × 2 fields)
        lookup_range_check_9_9_h_0: *const *const u32,
        lookup_range_check_9_9_h_1: *const *const u32,

        // range_check_19 (4 lookups × 1 field)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // range_check_19_b (4 lookups × 1 field)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // range_check_19_c (4 lookups × 1 field)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // range_check_19_d (3 lookups × 1 field)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // range_check_19_e (3 lookups × 1 field)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // range_check_19_f (3 lookups × 1 field)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // range_check_19_g (3 lookups × 1 field)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // range_check_19_h (4 lookups × 1 field)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // range_check_18 (1 lookup × 1 field)
        lookup_range_check_18_0: *const *const u32,

        // range_check_11 (1 lookup × 1 field)
        lookup_range_check_11_0: *const *const u32,

        // verify_instruction (1 lookup × 7 fields)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Opcode inputs
        generic_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_generic_opcode_interaction_traces(
        // Relation pointers (22 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_9_9: *mut c_void,
        range_check_9_9_b: *mut c_void,
        range_check_9_9_c: *mut c_void,
        range_check_9_9_d: *mut c_void,
        range_check_9_9_e: *mut c_void,
        range_check_9_9_f: *mut c_void,
        range_check_9_9_g: *mut c_void,
        range_check_9_9_h: *mut c_void,
        range_check_19: *mut c_void,
        range_check_19_b: *mut c_void,
        range_check_19_c: *mut c_void,
        range_check_19_d: *mut c_void,
        range_check_19_e: *mut c_void,
        range_check_19_f: *mut c_void,
        range_check_19_g: *mut c_void,
        range_check_19_h: *mut c_void,
        range_check_18: *mut c_void,
        range_check_11: *mut c_void,
        verify_instruction: *mut c_void,

        // All lookup data (67 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_9_9_0: *const *const u32,
        lookup_range_check_9_9_1: *const *const u32,
        lookup_range_check_9_9_2: *const *const u32,
        lookup_range_check_9_9_3: *const *const u32,
        lookup_range_check_9_9_b_0: *const *const u32,
        lookup_range_check_9_9_b_1: *const *const u32,
        lookup_range_check_9_9_b_2: *const *const u32,
        lookup_range_check_9_9_b_3: *const *const u32,
        lookup_range_check_9_9_c_0: *const *const u32,
        lookup_range_check_9_9_c_1: *const *const u32,
        lookup_range_check_9_9_c_2: *const *const u32,
        lookup_range_check_9_9_c_3: *const *const u32,
        lookup_range_check_9_9_d_0: *const *const u32,
        lookup_range_check_9_9_d_1: *const *const u32,
        lookup_range_check_9_9_d_2: *const *const u32,
        lookup_range_check_9_9_d_3: *const *const u32,
        lookup_range_check_9_9_e_0: *const *const u32,
        lookup_range_check_9_9_e_1: *const *const u32,
        lookup_range_check_9_9_e_2: *const *const u32,
        lookup_range_check_9_9_e_3: *const *const u32,
        lookup_range_check_9_9_f_0: *const *const u32,
        lookup_range_check_9_9_f_1: *const *const u32,
        lookup_range_check_9_9_f_2: *const *const u32,
        lookup_range_check_9_9_f_3: *const *const u32,
        lookup_range_check_9_9_g_0: *const *const u32,
        lookup_range_check_9_9_g_1: *const *const u32,
        lookup_range_check_9_9_h_0: *const *const u32,
        lookup_range_check_9_9_h_1: *const *const u32,
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jnz_opcode functions
    pub fn generate_jnz_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jnz_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jnz_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jnz_opcode_taken functions
    pub fn generate_jnz_opcode_taken_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jnz_opcode_taken_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jnz_opcode_taken_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode functions
    pub fn generate_jump_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_double_deref functions
    pub fn generate_jump_opcode_double_deref_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_double_deref_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_double_deref_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_rel functions
    pub fn generate_jump_opcode_rel_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_rel_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_rel_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_rel_imm functions
    pub fn generate_jump_opcode_rel_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_rel_imm_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_rel_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // mul_opcode functions (130 trace columns, 19 interaction columns)
    pub fn generate_mul_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_19 (4 lookups)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // Lookup data - range_check_19_b (4 lookups)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // Lookup data - range_check_19_c (4 lookups)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // Lookup data - range_check_19_d (3 lookups)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // Lookup data - range_check_19_e (3 lookups)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // Lookup data - range_check_19_f (3 lookups)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // Lookup data - range_check_19_g (3 lookups)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // Lookup data - range_check_19_h (4 lookups)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_19: *const *const u32,
        sub_component_inputs_range_check_19_b: *const *const u32,
        sub_component_inputs_range_check_19_c: *const *const u32,
        sub_component_inputs_range_check_19_d: *const *const u32,
        sub_component_inputs_range_check_19_e: *const *const u32,
        sub_component_inputs_range_check_19_f: *const *const u32,
        sub_component_inputs_range_check_19_g: *const *const u32,
        sub_component_inputs_range_check_19_h: *const *const u32,

        // Opcode inputs
        mul_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_opcode_interaction_traces(
        // Relation pointers (12 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_19: *mut c_void,
        range_check_19_b: *mut c_void,
        range_check_19_c: *mut c_void,
        range_check_19_d: *mut c_void,
        range_check_19_e: *mut c_void,
        range_check_19_f: *mut c_void,
        range_check_19_g: *mut c_void,
        range_check_19_h: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_19 (4 lookups)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // Lookup data - range_check_19_b (4 lookups)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // Lookup data - range_check_19_c (4 lookups)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // Lookup data - range_check_19_d (3 lookups)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // Lookup data - range_check_19_e (3 lookups)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // Lookup data - range_check_19_f (3 lookups)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // Lookup data - range_check_19_g (3 lookups)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // Lookup data - range_check_19_h (4 lookups)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // mul_opcode_small functions (37 trace columns, 6 interaction columns)
    pub fn generate_mul_opcode_small_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_11 (3 lookups)
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_11_1: *const *const u32,
        lookup_range_check_11_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_11: *const *const u32,

        // Opcode inputs
        mul_opcode_small_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_opcode_small_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_11: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_11 (3 lookups)
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_11_1: *const *const u32,
        lookup_range_check_11_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === QM_31_ADD_MUL_OPCODE ===
    pub fn generate_qm_31_add_mul_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (3 lookups)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_4_4_4_4: *const *const u32,

        // Opcode inputs
        qm_31_add_mul_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_qm_31_add_mul_opcode_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_4_4_4_4: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (3 lookups)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === RET_OPCODE ===
    pub fn generate_ret_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (2 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,

        // Lookup data - memory_id_to_big (2 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Opcode inputs
        ret_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_ret_opcode_interaction_traces(
        // Relation pointers (4 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        // Lookup data - memory_address_to_id (2 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,

        // Lookup data - memory_id_to_big (2 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === ADD_MOD_BUILTIN ===
    pub fn generate_add_mod_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_mod_builtin_interaction_traces(
        // Relation pointers (2 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === BITWISE_BUILTIN ===
    pub fn generate_bitwise_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (5 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,

        // Lookup data - memory_id_to_big (5 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,

        // Lookup data - verify_bitwise_xor_9 (27 lookups)
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,
        lookup_verify_bitwise_xor_9_2: *const *const u32,
        lookup_verify_bitwise_xor_9_3: *const *const u32,
        lookup_verify_bitwise_xor_9_4: *const *const u32,
        lookup_verify_bitwise_xor_9_5: *const *const u32,
        lookup_verify_bitwise_xor_9_6: *const *const u32,
        lookup_verify_bitwise_xor_9_7: *const *const u32,
        lookup_verify_bitwise_xor_9_8: *const *const u32,
        lookup_verify_bitwise_xor_9_9: *const *const u32,
        lookup_verify_bitwise_xor_9_10: *const *const u32,
        lookup_verify_bitwise_xor_9_11: *const *const u32,
        lookup_verify_bitwise_xor_9_12: *const *const u32,
        lookup_verify_bitwise_xor_9_13: *const *const u32,
        lookup_verify_bitwise_xor_9_14: *const *const u32,
        lookup_verify_bitwise_xor_9_15: *const *const u32,
        lookup_verify_bitwise_xor_9_16: *const *const u32,
        lookup_verify_bitwise_xor_9_17: *const *const u32,
        lookup_verify_bitwise_xor_9_18: *const *const u32,
        lookup_verify_bitwise_xor_9_19: *const *const u32,
        lookup_verify_bitwise_xor_9_20: *const *const u32,
        lookup_verify_bitwise_xor_9_21: *const *const u32,
        lookup_verify_bitwise_xor_9_22: *const *const u32,
        lookup_verify_bitwise_xor_9_23: *const *const u32,
        lookup_verify_bitwise_xor_9_24: *const *const u32,
        lookup_verify_bitwise_xor_9_25: *const *const u32,
        lookup_verify_bitwise_xor_9_26: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (1 lookup)
        lookup_verify_bitwise_xor_8_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_9: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_8: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_bitwise_builtin_interaction_traces(
        // Relation pointers (4 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        verify_bitwise_xor_9: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,

        // Lookup data - memory_address_to_id (5 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,

        // Lookup data - memory_id_to_big (5 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,

        // Lookup data - verify_bitwise_xor_9 (27 lookups)
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,
        lookup_verify_bitwise_xor_9_2: *const *const u32,
        lookup_verify_bitwise_xor_9_3: *const *const u32,
        lookup_verify_bitwise_xor_9_4: *const *const u32,
        lookup_verify_bitwise_xor_9_5: *const *const u32,
        lookup_verify_bitwise_xor_9_6: *const *const u32,
        lookup_verify_bitwise_xor_9_7: *const *const u32,
        lookup_verify_bitwise_xor_9_8: *const *const u32,
        lookup_verify_bitwise_xor_9_9: *const *const u32,
        lookup_verify_bitwise_xor_9_10: *const *const u32,
        lookup_verify_bitwise_xor_9_11: *const *const u32,
        lookup_verify_bitwise_xor_9_12: *const *const u32,
        lookup_verify_bitwise_xor_9_13: *const *const u32,
        lookup_verify_bitwise_xor_9_14: *const *const u32,
        lookup_verify_bitwise_xor_9_15: *const *const u32,
        lookup_verify_bitwise_xor_9_16: *const *const u32,
        lookup_verify_bitwise_xor_9_17: *const *const u32,
        lookup_verify_bitwise_xor_9_18: *const *const u32,
        lookup_verify_bitwise_xor_9_19: *const *const u32,
        lookup_verify_bitwise_xor_9_20: *const *const u32,
        lookup_verify_bitwise_xor_9_21: *const *const u32,
        lookup_verify_bitwise_xor_9_22: *const *const u32,
        lookup_verify_bitwise_xor_9_23: *const *const u32,
        lookup_verify_bitwise_xor_9_24: *const *const u32,
        lookup_verify_bitwise_xor_9_25: *const *const u32,
        lookup_verify_bitwise_xor_9_26: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (1 lookup)
        lookup_verify_bitwise_xor_8_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === MUL_MOD_BUILTIN ===
    pub fn generate_mul_mod_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Lookup data - range_check_12 (32 lookups)
        lookup_range_check_12_0: *const *const u32,
        lookup_range_check_12_1: *const *const u32,
        lookup_range_check_12_2: *const *const u32,
        lookup_range_check_12_3: *const *const u32,
        lookup_range_check_12_4: *const *const u32,
        lookup_range_check_12_5: *const *const u32,
        lookup_range_check_12_6: *const *const u32,
        lookup_range_check_12_7: *const *const u32,
        lookup_range_check_12_8: *const *const u32,
        lookup_range_check_12_9: *const *const u32,
        lookup_range_check_12_10: *const *const u32,
        lookup_range_check_12_11: *const *const u32,
        lookup_range_check_12_12: *const *const u32,
        lookup_range_check_12_13: *const *const u32,
        lookup_range_check_12_14: *const *const u32,
        lookup_range_check_12_15: *const *const u32,
        lookup_range_check_12_16: *const *const u32,
        lookup_range_check_12_17: *const *const u32,
        lookup_range_check_12_18: *const *const u32,
        lookup_range_check_12_19: *const *const u32,
        lookup_range_check_12_20: *const *const u32,
        lookup_range_check_12_21: *const *const u32,
        lookup_range_check_12_22: *const *const u32,
        lookup_range_check_12_23: *const *const u32,
        lookup_range_check_12_24: *const *const u32,
        lookup_range_check_12_25: *const *const u32,
        lookup_range_check_12_26: *const *const u32,
        lookup_range_check_12_27: *const *const u32,
        lookup_range_check_12_28: *const *const u32,
        lookup_range_check_12_29: *const *const u32,
        lookup_range_check_12_30: *const *const u32,
        lookup_range_check_12_31: *const *const u32,

        // Lookup data - range_check_18 (62 lookups)
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_18_1: *const *const u32,
        lookup_range_check_18_2: *const *const u32,
        lookup_range_check_18_3: *const *const u32,
        lookup_range_check_18_4: *const *const u32,
        lookup_range_check_18_5: *const *const u32,
        lookup_range_check_18_6: *const *const u32,
        lookup_range_check_18_7: *const *const u32,
        lookup_range_check_18_8: *const *const u32,
        lookup_range_check_18_9: *const *const u32,
        lookup_range_check_18_10: *const *const u32,
        lookup_range_check_18_11: *const *const u32,
        lookup_range_check_18_12: *const *const u32,
        lookup_range_check_18_13: *const *const u32,
        lookup_range_check_18_14: *const *const u32,
        lookup_range_check_18_15: *const *const u32,
        lookup_range_check_18_16: *const *const u32,
        lookup_range_check_18_17: *const *const u32,
        lookup_range_check_18_18: *const *const u32,
        lookup_range_check_18_19: *const *const u32,
        lookup_range_check_18_20: *const *const u32,
        lookup_range_check_18_21: *const *const u32,
        lookup_range_check_18_22: *const *const u32,
        lookup_range_check_18_23: *const *const u32,
        lookup_range_check_18_24: *const *const u32,
        lookup_range_check_18_25: *const *const u32,
        lookup_range_check_18_26: *const *const u32,
        lookup_range_check_18_27: *const *const u32,
        lookup_range_check_18_28: *const *const u32,
        lookup_range_check_18_29: *const *const u32,
        lookup_range_check_18_30: *const *const u32,
        lookup_range_check_18_31: *const *const u32,
        lookup_range_check_18_32: *const *const u32,
        lookup_range_check_18_33: *const *const u32,
        lookup_range_check_18_34: *const *const u32,
        lookup_range_check_18_35: *const *const u32,
        lookup_range_check_18_36: *const *const u32,
        lookup_range_check_18_37: *const *const u32,
        lookup_range_check_18_38: *const *const u32,
        lookup_range_check_18_39: *const *const u32,
        lookup_range_check_18_40: *const *const u32,
        lookup_range_check_18_41: *const *const u32,
        lookup_range_check_18_42: *const *const u32,
        lookup_range_check_18_43: *const *const u32,
        lookup_range_check_18_44: *const *const u32,
        lookup_range_check_18_45: *const *const u32,
        lookup_range_check_18_46: *const *const u32,
        lookup_range_check_18_47: *const *const u32,
        lookup_range_check_18_48: *const *const u32,
        lookup_range_check_18_49: *const *const u32,
        lookup_range_check_18_50: *const *const u32,
        lookup_range_check_18_51: *const *const u32,
        lookup_range_check_18_52: *const *const u32,
        lookup_range_check_18_53: *const *const u32,
        lookup_range_check_18_54: *const *const u32,
        lookup_range_check_18_55: *const *const u32,
        lookup_range_check_18_56: *const *const u32,
        lookup_range_check_18_57: *const *const u32,
        lookup_range_check_18_58: *const *const u32,
        lookup_range_check_18_59: *const *const u32,
        lookup_range_check_18_60: *const *const u32,
        lookup_range_check_18_61: *const *const u32,

        // Lookup data - range_check_3_6_6_3 (40 lookups)
        lookup_range_check_3_6_6_3_0: *const *const u32,
        lookup_range_check_3_6_6_3_1: *const *const u32,
        lookup_range_check_3_6_6_3_2: *const *const u32,
        lookup_range_check_3_6_6_3_3: *const *const u32,
        lookup_range_check_3_6_6_3_4: *const *const u32,
        lookup_range_check_3_6_6_3_5: *const *const u32,
        lookup_range_check_3_6_6_3_6: *const *const u32,
        lookup_range_check_3_6_6_3_7: *const *const u32,
        lookup_range_check_3_6_6_3_8: *const *const u32,
        lookup_range_check_3_6_6_3_9: *const *const u32,
        lookup_range_check_3_6_6_3_10: *const *const u32,
        lookup_range_check_3_6_6_3_11: *const *const u32,
        lookup_range_check_3_6_6_3_12: *const *const u32,
        lookup_range_check_3_6_6_3_13: *const *const u32,
        lookup_range_check_3_6_6_3_14: *const *const u32,
        lookup_range_check_3_6_6_3_15: *const *const u32,
        lookup_range_check_3_6_6_3_16: *const *const u32,
        lookup_range_check_3_6_6_3_17: *const *const u32,
        lookup_range_check_3_6_6_3_18: *const *const u32,
        lookup_range_check_3_6_6_3_19: *const *const u32,
        lookup_range_check_3_6_6_3_20: *const *const u32,
        lookup_range_check_3_6_6_3_21: *const *const u32,
        lookup_range_check_3_6_6_3_22: *const *const u32,
        lookup_range_check_3_6_6_3_23: *const *const u32,
        lookup_range_check_3_6_6_3_24: *const *const u32,
        lookup_range_check_3_6_6_3_25: *const *const u32,
        lookup_range_check_3_6_6_3_26: *const *const u32,
        lookup_range_check_3_6_6_3_27: *const *const u32,
        lookup_range_check_3_6_6_3_28: *const *const u32,
        lookup_range_check_3_6_6_3_29: *const *const u32,
        lookup_range_check_3_6_6_3_30: *const *const u32,
        lookup_range_check_3_6_6_3_31: *const *const u32,
        lookup_range_check_3_6_6_3_32: *const *const u32,
        lookup_range_check_3_6_6_3_33: *const *const u32,
        lookup_range_check_3_6_6_3_34: *const *const u32,
        lookup_range_check_3_6_6_3_35: *const *const u32,
        lookup_range_check_3_6_6_3_36: *const *const u32,
        lookup_range_check_3_6_6_3_37: *const *const u32,
        lookup_range_check_3_6_6_3_38: *const *const u32,
        lookup_range_check_3_6_6_3_39: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_12: *const *const u32,
        sub_component_inputs_range_check_18: *const *const u32,
        sub_component_inputs_range_check_3_6_6_3: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_mod_builtin_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_12: *mut c_void,
        range_check_18: *mut c_void,
        range_check_3_6_6_3: *mut c_void,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Lookup data - range_check_12 (32 lookups)
        lookup_range_check_12_0: *const *const u32,
        lookup_range_check_12_1: *const *const u32,
        lookup_range_check_12_2: *const *const u32,
        lookup_range_check_12_3: *const *const u32,
        lookup_range_check_12_4: *const *const u32,
        lookup_range_check_12_5: *const *const u32,
        lookup_range_check_12_6: *const *const u32,
        lookup_range_check_12_7: *const *const u32,
        lookup_range_check_12_8: *const *const u32,
        lookup_range_check_12_9: *const *const u32,
        lookup_range_check_12_10: *const *const u32,
        lookup_range_check_12_11: *const *const u32,
        lookup_range_check_12_12: *const *const u32,
        lookup_range_check_12_13: *const *const u32,
        lookup_range_check_12_14: *const *const u32,
        lookup_range_check_12_15: *const *const u32,
        lookup_range_check_12_16: *const *const u32,
        lookup_range_check_12_17: *const *const u32,
        lookup_range_check_12_18: *const *const u32,
        lookup_range_check_12_19: *const *const u32,
        lookup_range_check_12_20: *const *const u32,
        lookup_range_check_12_21: *const *const u32,
        lookup_range_check_12_22: *const *const u32,
        lookup_range_check_12_23: *const *const u32,
        lookup_range_check_12_24: *const *const u32,
        lookup_range_check_12_25: *const *const u32,
        lookup_range_check_12_26: *const *const u32,
        lookup_range_check_12_27: *const *const u32,
        lookup_range_check_12_28: *const *const u32,
        lookup_range_check_12_29: *const *const u32,
        lookup_range_check_12_30: *const *const u32,
        lookup_range_check_12_31: *const *const u32,

        // Lookup data - range_check_18 (62 lookups)
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_18_1: *const *const u32,
        lookup_range_check_18_2: *const *const u32,
        lookup_range_check_18_3: *const *const u32,
        lookup_range_check_18_4: *const *const u32,
        lookup_range_check_18_5: *const *const u32,
        lookup_range_check_18_6: *const *const u32,
        lookup_range_check_18_7: *const *const u32,
        lookup_range_check_18_8: *const *const u32,
        lookup_range_check_18_9: *const *const u32,
        lookup_range_check_18_10: *const *const u32,
        lookup_range_check_18_11: *const *const u32,
        lookup_range_check_18_12: *const *const u32,
        lookup_range_check_18_13: *const *const u32,
        lookup_range_check_18_14: *const *const u32,
        lookup_range_check_18_15: *const *const u32,
        lookup_range_check_18_16: *const *const u32,
        lookup_range_check_18_17: *const *const u32,
        lookup_range_check_18_18: *const *const u32,
        lookup_range_check_18_19: *const *const u32,
        lookup_range_check_18_20: *const *const u32,
        lookup_range_check_18_21: *const *const u32,
        lookup_range_check_18_22: *const *const u32,
        lookup_range_check_18_23: *const *const u32,
        lookup_range_check_18_24: *const *const u32,
        lookup_range_check_18_25: *const *const u32,
        lookup_range_check_18_26: *const *const u32,
        lookup_range_check_18_27: *const *const u32,
        lookup_range_check_18_28: *const *const u32,
        lookup_range_check_18_29: *const *const u32,
        lookup_range_check_18_30: *const *const u32,
        lookup_range_check_18_31: *const *const u32,
        lookup_range_check_18_32: *const *const u32,
        lookup_range_check_18_33: *const *const u32,
        lookup_range_check_18_34: *const *const u32,
        lookup_range_check_18_35: *const *const u32,
        lookup_range_check_18_36: *const *const u32,
        lookup_range_check_18_37: *const *const u32,
        lookup_range_check_18_38: *const *const u32,
        lookup_range_check_18_39: *const *const u32,
        lookup_range_check_18_40: *const *const u32,
        lookup_range_check_18_41: *const *const u32,
        lookup_range_check_18_42: *const *const u32,
        lookup_range_check_18_43: *const *const u32,
        lookup_range_check_18_44: *const *const u32,
        lookup_range_check_18_45: *const *const u32,
        lookup_range_check_18_46: *const *const u32,
        lookup_range_check_18_47: *const *const u32,
        lookup_range_check_18_48: *const *const u32,
        lookup_range_check_18_49: *const *const u32,
        lookup_range_check_18_50: *const *const u32,
        lookup_range_check_18_51: *const *const u32,
        lookup_range_check_18_52: *const *const u32,
        lookup_range_check_18_53: *const *const u32,
        lookup_range_check_18_54: *const *const u32,
        lookup_range_check_18_55: *const *const u32,
        lookup_range_check_18_56: *const *const u32,
        lookup_range_check_18_57: *const *const u32,
        lookup_range_check_18_58: *const *const u32,
        lookup_range_check_18_59: *const *const u32,
        lookup_range_check_18_60: *const *const u32,
        lookup_range_check_18_61: *const *const u32,

        // Lookup data - range_check_3_6_6_3 (40 lookups)
        lookup_range_check_3_6_6_3_0: *const *const u32,
        lookup_range_check_3_6_6_3_1: *const *const u32,
        lookup_range_check_3_6_6_3_2: *const *const u32,
        lookup_range_check_3_6_6_3_3: *const *const u32,
        lookup_range_check_3_6_6_3_4: *const *const u32,
        lookup_range_check_3_6_6_3_5: *const *const u32,
        lookup_range_check_3_6_6_3_6: *const *const u32,
        lookup_range_check_3_6_6_3_7: *const *const u32,
        lookup_range_check_3_6_6_3_8: *const *const u32,
        lookup_range_check_3_6_6_3_9: *const *const u32,
        lookup_range_check_3_6_6_3_10: *const *const u32,
        lookup_range_check_3_6_6_3_11: *const *const u32,
        lookup_range_check_3_6_6_3_12: *const *const u32,
        lookup_range_check_3_6_6_3_13: *const *const u32,
        lookup_range_check_3_6_6_3_14: *const *const u32,
        lookup_range_check_3_6_6_3_15: *const *const u32,
        lookup_range_check_3_6_6_3_16: *const *const u32,
        lookup_range_check_3_6_6_3_17: *const *const u32,
        lookup_range_check_3_6_6_3_18: *const *const u32,
        lookup_range_check_3_6_6_3_19: *const *const u32,
        lookup_range_check_3_6_6_3_20: *const *const u32,
        lookup_range_check_3_6_6_3_21: *const *const u32,
        lookup_range_check_3_6_6_3_22: *const *const u32,
        lookup_range_check_3_6_6_3_23: *const *const u32,
        lookup_range_check_3_6_6_3_24: *const *const u32,
        lookup_range_check_3_6_6_3_25: *const *const u32,
        lookup_range_check_3_6_6_3_26: *const *const u32,
        lookup_range_check_3_6_6_3_27: *const *const u32,
        lookup_range_check_3_6_6_3_28: *const *const u32,
        lookup_range_check_3_6_6_3_29: *const *const u32,
        lookup_range_check_3_6_6_3_30: *const *const u32,
        lookup_range_check_3_6_6_3_31: *const *const u32,
        lookup_range_check_3_6_6_3_32: *const *const u32,
        lookup_range_check_3_6_6_3_33: *const *const u32,
        lookup_range_check_3_6_6_3_34: *const *const u32,
        lookup_range_check_3_6_6_3_35: *const *const u32,
        lookup_range_check_3_6_6_3_36: *const *const u32,
        lookup_range_check_3_6_6_3_37: *const *const u32,
        lookup_range_check_3_6_6_3_38: *const *const u32,
        lookup_range_check_3_6_6_3_39: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === PEDERSEN_BUILTIN ===
    pub fn gen_pedersen_builtin_trace(
        traces: *const *const u32,
        n_trace_columns: u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - partial_ec_mul (8 lookups)
        lookup_partial_ec_mul_0: *const *const u32,
        lookup_partial_ec_mul_1: *const *const u32,
        lookup_partial_ec_mul_2: *const *const u32,
        lookup_partial_ec_mul_3: *const *const u32,
        lookup_partial_ec_mul_4: *const *const u32,
        lookup_partial_ec_mul_5: *const *const u32,
        lookup_partial_ec_mul_6: *const *const u32,
        lookup_partial_ec_mul_7: *const *const u32,

        // Lookup data - range_check_5_4 (2 lookups)
        lookup_range_check_5_4_0: *const *const u32,
        lookup_range_check_5_4_1: *const *const u32,

        // Lookup data - range_check_8 (4 lookups)
        lookup_range_check_8_0: *const *const u32,
        lookup_range_check_8_1: *const *const u32,
        lookup_range_check_8_2: *const *const u32,
        lookup_range_check_8_3: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        // Pre-computed EC columns (columns 66-350)
        precomputed_ec_columns: *const *const u32,
        n_precomputed_ec_columns: u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn gen_pedersen_builtin_interaction_trace(
        interaction_trace: *const *const u32,
        n_interaction_columns: u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - partial_ec_mul (8 lookups)
        lookup_partial_ec_mul_0: *const *const u32,
        lookup_partial_ec_mul_1: *const *const u32,
        lookup_partial_ec_mul_2: *const *const u32,
        lookup_partial_ec_mul_3: *const *const u32,
        lookup_partial_ec_mul_4: *const *const u32,
        lookup_partial_ec_mul_5: *const *const u32,
        lookup_partial_ec_mul_6: *const *const u32,
        lookup_partial_ec_mul_7: *const *const u32,

        // Lookup data - range_check_5_4 (2 lookups)
        lookup_range_check_5_4_0: *const *const u32,
        lookup_range_check_5_4_1: *const *const u32,

        // Lookup data - range_check_8 (4 lookups)
        lookup_range_check_8_0: *const *const u32,
        lookup_range_check_8_1: *const *const u32,
        lookup_range_check_8_2: *const *const u32,
        lookup_range_check_8_3: *const *const u32,

        // Lookup elements (relations)
        memory_address_to_id: *const u32,
        memory_id_to_big: *const u32,
        partial_ec_mul: *const u32,
        range_check_5_4: *const u32,
        range_check_8: *const u32,

        claimed_sum: *const u32,
        n_rows: u32,
        log_size: u32,
    );

    // Pedersen table management - upload and free the ~1.8GB PEDERSEN_TABLE for GPU EC operations
    pub fn pedersen_table_init(
        columns: *const *const u32,  // Array of 56 column pointers (28 for x, 28 for y)
        n_rows: u32,                  // Number of rows in the table (~8M)
    );

    pub fn pedersen_table_free();

    // Full GPU trace generation for pedersen_builtin (computes partial_ec_mul on GPU)
    pub fn gen_pedersen_builtin_trace_full_gpu(
        traces: *const *const u32,
        n_trace_columns: u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - partial_ec_mul (8 lookups)
        lookup_partial_ec_mul_0: *const *const u32,
        lookup_partial_ec_mul_1: *const *const u32,
        lookup_partial_ec_mul_2: *const *const u32,
        lookup_partial_ec_mul_3: *const *const u32,
        lookup_partial_ec_mul_4: *const *const u32,
        lookup_partial_ec_mul_5: *const *const u32,
        lookup_partial_ec_mul_6: *const *const u32,
        lookup_partial_ec_mul_7: *const *const u32,

        // Lookup data - range_check_5_4 (2 lookups)
        lookup_range_check_5_4_0: *const *const u32,
        lookup_range_check_5_4_1: *const *const u32,

        // Lookup data - range_check_8 (4 lookups)
        lookup_range_check_8_0: *const *const u32,
        lookup_range_check_8_1: *const *const u32,
        lookup_range_check_8_2: *const *const u32,
        lookup_range_check_8_3: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    // === POSEIDON_BUILTIN ===
    pub fn gen_poseidon_builtin_trace(
        traces: *const *const u32,
        log_size: u32,
        segment_start: u32,
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        // Lookup data - memory_address_to_id (6 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,

        // Lookup data - memory_id_to_big (6 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,

        // Lookup data - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
        lookup_range_check_3_3_3_3_3_0: *const *const u32,
        lookup_range_check_3_3_3_3_3_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (6 lookups, 4 elements each)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,
        lookup_range_check_4_4_4_4_3: *const *const u32,
        lookup_range_check_4_4_4_4_4: *const *const u32,
        lookup_range_check_4_4_4_4_5: *const *const u32,

        // Lookup data - range_check_4_4 (3 lookups, 2 elements each)
        lookup_range_check_4_4_0: *const *const u32,
        lookup_range_check_4_4_1: *const *const u32,
        lookup_range_check_4_4_2: *const *const u32,

        // Lookup data - poseidon_full_round_chain (2 lookups, 32 elements each)
        lookup_poseidon_full_round_chain_0: *const *const u32,
        lookup_poseidon_full_round_chain_1: *const *const u32,

        // Base trace cols 120-283 (164 columns) for interaction kernels
        lookup_base_trace_cols: *const *const u32,
    );

    pub fn gen_poseidon_builtin_interaction_trace(
        interaction_trace: *const *const u32,
        log_size: u32,

        // Lookup elements (relations)
        memory_address_to_id_relation: *const u32,
        memory_id_to_big_relation: *const u32,
        poseidon_full_round_chain_relation: *const u32,
        range_check_felt_252_width_27_relation: *const u32,
        cube_252_relation: *const u32,
        range_check_3_3_3_3_3_relation: *const u32,
        range_check_4_4_4_4_relation: *const u32,
        range_check_4_4_relation: *const u32,
        poseidon_3_partial_rounds_chain_relation: *const u32,

        // Base trace columns for reconstructing all lookup data
        base_trace: *const *const u32,

        // Lookup data - memory_address_to_id (6 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,

        // Lookup data - memory_id_to_big (6 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,

        // Lookup data - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
        lookup_range_check_3_3_3_3_3_0: *const *const u32,
        lookup_range_check_3_3_3_3_3_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (6 lookups, 4 elements each)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,
        lookup_range_check_4_4_4_4_3: *const *const u32,
        lookup_range_check_4_4_4_4_4: *const *const u32,
        lookup_range_check_4_4_4_4_5: *const *const u32,

        // Lookup data - range_check_4_4 (3 lookups, 2 elements each)
        lookup_range_check_4_4_0: *const *const u32,
        lookup_range_check_4_4_1: *const *const u32,
        lookup_range_check_4_4_2: *const *const u32,

        // Lookup data - poseidon_full_round_chain (2 lookups, 32 elements each)
        lookup_poseidon_full_round_chain_0: *const *const u32,
        lookup_poseidon_full_round_chain_1: *const *const u32,

        // Base trace cols 120-283 (164 columns) for interaction kernels
        lookup_base_trace_cols: *const *const u32,

        claimed_sum: *const u32,
    );

    // === RANGE_CHECK_BUILTIN_BITS_96 ===
    pub fn generate_range_check_builtin_bits_96_traces(
        traces: *const *const u32,

        // Lookup data - 1 MemoryAddressToId lookup (2 elements)
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup (29 elements)
        lookup_memory_id_to_big_0: *const *const u32,

        // Lookup data - 1 RangeCheck_6 lookup (1 element)
        lookup_range_check_6_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_6: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_range_check_builtin_bits_96_interaction_traces(
        // Relation pointers (3 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_6: *mut c_void,

        // Lookup data - 1 MemoryAddressToId lookup
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup
        lookup_memory_id_to_big_0: *const *const u32,

        // Lookup data - 1 RangeCheck_6 lookup
        lookup_range_check_6_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === RANGE_CHECK_BUILTIN_BITS_128 ===
    pub fn generate_range_check_builtin_bits_128_traces(
        traces: *const *const u32,

        // Lookup data - 1 MemoryAddressToId lookup (2 elements)
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup (29 elements)
        lookup_memory_id_to_big_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_range_check_builtin_bits_128_interaction_traces(
        // Relation pointers (2 relations - no range_check_6)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,

        // Lookup data - 1 MemoryAddressToId lookup
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup
        lookup_memory_id_to_big_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

}