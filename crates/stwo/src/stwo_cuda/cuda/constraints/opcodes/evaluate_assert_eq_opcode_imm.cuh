#ifndef EVALUATE_ASSERT_EQ_OPCODE_IMM_H
#define EVALUATE_ASSERT_EQ_OPCODE_IMM_H


#include "fields.cuh"
#include "relations.cuh"

struct AssertEqImmClaim {
    unsigned log_size;
};

struct AssertEqImmEval {
    unsigned eval_id;
    AssertEqImmClaim Claim;
    VerifyInstruction verify_instruction_lookup_elements;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    Opcodes opcode_lookup_elements;
};

extern "C"
void evaluate_assert_eq_opcode_imm(
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

#endif // EVALUATE_ASSERT_EQ_OPCODE_IMM_H