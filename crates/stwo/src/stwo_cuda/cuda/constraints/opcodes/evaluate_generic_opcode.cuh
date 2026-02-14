#ifndef EVALUATE_GENERIC_OPCODE_CUH
#define EVALUATE_GENERIC_OPCODE_CUH

#include "fields.cuh"
#include "relations.cuh"

// Generic opcode component has 244 trace columns
// This is a complex opcode that handles generic Cairo VM instructions
// with four subroutines: DecodeGenericInstruction, EvalOperands, HandleOpcodes, UpdateRegisters

// Component Eval struct matching Rust definition
// Must match memory layout of generic_opcode::Eval in generic_opcode.rs
struct GenericOpcode_Eval {
    uint32_t eval_id;
    uint32_t log_size;  // from Claim
    VerifyInstruction verify_instruction_lookup_elements;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    MemoryIdToBig memory_id_to_big_lookup_elements;
    RangeCheck_9_9 range_check_9_9_lookup_elements;
    RangeCheck_9_9_B range_check_9_9_b_lookup_elements;
    RangeCheck_9_9_C range_check_9_9_c_lookup_elements;
    RangeCheck_9_9_D range_check_9_9_d_lookup_elements;
    RangeCheck_9_9_E range_check_9_9_e_lookup_elements;
    RangeCheck_9_9_F range_check_9_9_f_lookup_elements;
    RangeCheck_9_9_G range_check_9_9_g_lookup_elements;
    RangeCheck_9_9_H range_check_9_9_h_lookup_elements;
    RangeCheck_19_H range_check_19_h_lookup_elements;
    RangeCheck_19 range_check_19_lookup_elements;
    RangeCheck_19_B range_check_19_b_lookup_elements;
    RangeCheck_19_C range_check_19_c_lookup_elements;
    RangeCheck_19_D range_check_19_d_lookup_elements;
    RangeCheck_19_E range_check_19_e_lookup_elements;
    RangeCheck_19_F range_check_19_f_lookup_elements;
    RangeCheck_19_G range_check_19_g_lookup_elements;
    RangeCheck_18 range_check_18_lookup_elements;
    RangeCheck_11 range_check_11_lookup_elements;
    Opcodes opcodes_lookup_elements;
};

extern "C"
void evaluate_generic_opcode(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
);

#endif // EVALUATE_GENERIC_OPCODE_CUH
