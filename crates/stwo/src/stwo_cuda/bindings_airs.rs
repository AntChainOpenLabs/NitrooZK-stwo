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

}