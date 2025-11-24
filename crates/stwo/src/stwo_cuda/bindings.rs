use std::ffi::c_void;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};
// use crate::stwo_cuda::mem_pool; // DEPRECATED: No longer needed

#[repr(C)]
pub struct CudaSecureField {
    a: BaseField,
    b: BaseField,
    c: BaseField,
    d: BaseField,
}

impl CudaSecureField {
    pub fn zero() -> Self {
        Self {
            a: BaseField::from(0),
            b: BaseField::from(0),
            c: BaseField::from(0),
            d: BaseField::from(0),
        }
    }
}

impl From<SecureField> for CudaSecureField {
    fn from(value: SecureField) -> Self {
        Self {
            a: value.0 .0,
            b: value.0 .1,
            c: value.1 .0,
            d: value.1 .1,
        }
    }
}

impl From<CudaSecureField> for SecureField {
    fn from(value: CudaSecureField) -> Self {
        SecureField::from_m31(value.a, value.b, value.c, value.d)
    }
}

// This is needed since `CirclePoint<BaseField>` is not FFI safe.
#[repr(C)]
pub struct CirclePointBaseField {
    x: BaseField,
    y: BaseField,
}

#[repr(C)]
pub struct LayerIndexPair {
    pub layer_idx: u32,
    pub hash_idx: u32,
}

impl From<CirclePoint<BaseField>> for CirclePointBaseField {
    fn from(value: CirclePoint<BaseField>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[repr(C)]
pub(crate) struct CirclePointSecureField {
    x: CudaSecureField,
    y: CudaSecureField,
}

impl From<CirclePoint<SecureField>> for CirclePointSecureField {
    fn from(value: CirclePoint<SecureField>) -> Self {
        Self {
            x: CudaSecureField::from(value.x),
            y: CudaSecureField::from(value.y),
        }
    }
}

#[link(name = "stwo_cuda")]
extern "C" {
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );

    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device_offset(
        from: *const u32,
        dst: *const u32,
        size: u32,
        offset: u32,
    );

    pub fn cuda_malloc_uint32_t(size: u32) -> *const u32;

    pub fn cuda_set_uint32_t(device_ptr: *const c_void, index: usize, val: u32);

    pub fn cuda_get_uint32_t(device_ptr: *const c_void, index: usize) -> u32;

    pub fn cuda_increase_at(device_ptr: *const c_void, addr: u32);

    pub fn cuda_get_secure_field(device_ptr: *const c_void, index: usize) -> CudaSecureField;

    pub fn cuda_malloc_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32;

    pub fn cuda_alloc_zeroes_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_free_memory(device_ptr: *const c_void);

    pub fn bit_reverse_base_field(array: *const u32, size: usize);

    pub fn bit_reverse_secure_field(array: *const u32, size: usize);

    pub fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize);

    // pub fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);

    pub fn sort_values_and_permute_with_bit_reverse_order(
        from: *const u32,
        size: usize,
    ) -> *const u32;

    pub fn precompute_twiddles(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;

    pub fn evaluate_columns(
        eval_domain_sizes: *const u32,
        values: *const *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        number_of_columns: u32,
        column_sizes: *const u32,
    );

    pub fn eval_at_point(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
    ) -> CudaSecureField;

    pub fn fold_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn accumulate(size: u32, left_columns: *const *const u32, right_columns: *const *const u32);

    pub fn commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    pub fn copy_blake_2s_hash_vec_from_host_to_device(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;

    pub fn copy_blake_2s_hash_vec_from_device_to_host(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
        size: usize,
    );

    pub fn copy_blake_2s_hash_vec_from_device_to_device(
        from: *const Blake2sHash,
        dst: *const Blake2sHash,
        size: usize,
    );

    pub fn cuda_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *const Blake2sHash,
        index: usize,
    );

    pub fn cuda_batch_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        indices: *const u32,
        n_indices: u32,
    );

    pub fn cuda_multi_layer_batch_get_blake_2s_hash(
        layer_device_ptrs: *const *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        pairs: *const LayerIndexPair,
        n_pairs: u32,
    );

    pub fn copy_device_pointer_vec_from_host_to_device(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;

    pub fn accumulate_quotients(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        columns: *const *const u32,
        number_of_columns: usize,
        random_coeff: CudaSecureField,
        sample_points: *const u32,
        sample_columns_indexes: *const u32,
        sample_columns_indexes_size: u32,
        sample_column_values: *const CudaSecureField,
        sample_column_and_values_sizes: *const u32,
        sample_size: u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
        flattened_line_coeffs_size: u32,
    );

    pub fn gen_eq_evals(
        v: CudaSecureField,
        y: *const CudaSecureField,
        y_size: u32,
        evals: *const CudaSecureField,
        evals_size: u32,
    );

    pub fn fix_first_variable_base_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    pub fn fix_first_variable_secure_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );


    pub fn generate_wide_fibonacci_trace(
        input_a: *const u32,
        input_b: *const u32,
        input_len: u32,
        traces: *const *const u32,
        trace_len: u32,
        n_columns: u32,
    );

    pub fn generate_poseidon_traces(
        traces: *const *const u32,
        lookup_init: *const *const u32,
        lookup_final: *const *const u32,
        trace_log_len: u32,
    );

    pub fn generate_poseidon_interaction_traces(
        lookup_element: *mut c_void,
        lookup_init: *const *const u32,
        lookup_final: *const *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Assert EQ FP IMM trace generation
    pub fn generate_assert_eq_fp_imm_traces(
        trace_columns: *const *const u32,
        trace_columns_len: u32,
        registers_lookups: *const *const u32,
        registers_lookups_len: u32,
        memory_lookups: *const *const u32,
        memory_lookups_len: u32,
        range_check_20_lookups: *const *const u32,
        range_check_20_lookups_len: u32,
        inputs: *const c_void,  // AssertEqFpImmInput*
        inputs_len: u32,
        data_accesses: *const c_void,  // DataAccess*
        data_accesses_len: u32,
        log_size: u32,
        non_padded_length: u32,
    );

    pub fn evaluate_constraint_quotients_on_domain(
        quotients_0: *const u32,
        quotients_1: *const u32,
        quotients_2: *const u32,
        quotients_3: *const u32,
        trace0_evaluations: *const *const u32,
        trace0_evaluations_len: u32,
        trace1_evaluations: *const *const u32,
        trace1_evaluations_len: u32,
        trace2_evaluations: *const *const u32,
        trace2_evaluations_len: u32,
        random_coeff_powers: *const u32,
        denominator_inverses: *const u32,
        domain_log_size: u32,
        eval_domain_log_size: u32,
        number_of_columns: u32,
        logup_counts: u32,
        eval: *mut c_void,
        cumsum_shift: CudaSecureField,
        should_accumulate: bool,
        use_assert_evaluator: bool,
    );

    pub fn ntt_n2b_native_batch(
        value: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        start_stage: u32,
        end_stage: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_b2n_column(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_n2b_columns(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    // pub fn inclusive_prefix_sum(
    //     device_bit_rev_circle_domain_evals:  *const u32,
    //     len: u32,
    // );

    // Poseidon252 CUDA acceleration functions
    // Note: FieldElement252 is represented as 32 bytes (8 x u32)
    // Note: Poseidon252Hash is 32-byte struct, equivalent to [u8; 32]
    pub fn cuda_malloc_poseidon252_hash(size: usize) -> *mut [u8; 32];

    pub fn cuda_alloc_zeroes_poseidon252_hash(size: usize) -> *mut [u8; 32];

    pub fn copy_poseidon252_hash_vec_from_host_to_device(
        from: *const [u8; 32],
        size: usize,
    ) -> *mut [u8; 32];

    pub fn copy_poseidon252_hash_vec_from_device_to_host(
        from: *const [u8; 32],
        to: *mut [u8; 32],
        size: usize,
    );

    pub fn copy_poseidon252_hash_vec_from_device_to_device(
        from: *const [u8; 32],
        dst: *mut [u8; 32],
        size: usize,
    );

    pub fn cuda_get_poseidon252_hash(
        device_ptr: *const [u8; 32],
        host_ptr: *mut [u8; 32],
        index: usize,
    );

    pub fn cuda_set_poseidon252_hash(
        device_ptr: *mut [u8; 32],
        index: usize,
        value: *const [u8; 32],
    );

    // Hybrid Poseidon252 Merkle functions that use GPU for data processing
    // and CPU for verified hashing
    // GPU-only aliases matching Blake2s interface
    pub fn poseidon252_commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut [u8; 32],
    );

    pub fn poseidon252_commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const [u8; 32],
        result: *mut [u8; 32],
    );


}
