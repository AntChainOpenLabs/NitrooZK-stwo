#ifndef EVALUATE_JUMP_OPCODE_DOUBLE_DEREF_H
#define EVALUATE_JUMP_OPCODE_DOUBLE_DEREF_H

#include "constraints/relations.cuh"

// Claim structure matching the Rust Claim struct
struct JumpOpcodeDoubleDeref_Claim {
    unsigned log_size;
};

// Structure to hold lookup elements for jump_opcode_double_deref component
struct JumpOpcodeDoubleDeref_Eval {
    unsigned eval_id;
    JumpOpcodeDoubleDeref_Claim claim;  // Must match Rust struct layout
    VerifyInstruction verify_instruction_lookup_elements;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    MemoryIdToBig memory_id_to_big_lookup_elements;
    Opcodes opcode_lookup_elements;
};

// External function declaration for CUDA kernel
extern "C" void evaluate_jump_opcode_double_deref(
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
    bool use_assert_evaluator
);

#endif // EVALUATE_JUMP_OPCODE_DOUBLE_DEREF_H
