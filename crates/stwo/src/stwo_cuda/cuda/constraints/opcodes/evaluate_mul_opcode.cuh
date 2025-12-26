#ifndef EVALUATE_MUL_OPCODE_H
#define EVALUATE_MUL_OPCODE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

struct MulOpcode_Claim {
    unsigned log_size;
};

struct MulOpcode_Eval {
    unsigned eval_id;
    MulOpcode_Claim Claim;
    VerifyInstruction verify_instruction_lookup_elements;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    MemoryIdToBig memory_id_to_big_lookup_elements;
    RangeCheck_19_H range_check_19_h_lookup_elements;
    RangeCheck_19 range_check_19_lookup_elements;
    RangeCheck_19_B range_check_19_b_lookup_elements;
    RangeCheck_19_C range_check_19_c_lookup_elements;
    RangeCheck_19_D range_check_19_d_lookup_elements;
    RangeCheck_19_E range_check_19_e_lookup_elements;
    RangeCheck_19_F range_check_19_f_lookup_elements;
    RangeCheck_19_G range_check_19_g_lookup_elements;
    Opcodes opcode_lookup_elements;
};

extern "C"
void evaluate_mul_opcode(
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

#endif // EVALUATE_MUL_OPCODE_H
