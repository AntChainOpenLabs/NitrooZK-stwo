#ifndef EVALUATE_BLAKE_COMPRESS_OPCODE_CONSTRAINT_H
#define EVALUATE_BLAKE_COMPRESS_OPCODE_CONSTRAINT_H


#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

struct BlakeCompressOpcode_Claim {
    unsigned log_size;
};

struct BlakeCompressOpcode_Eval {
    unsigned eval_id;
    BlakeCompressOpcode_Claim Claim;
    VerifyInstruction verify_instruction_lookup_elements;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    MemoryIdToBig memory_id_to_big_lookup_elements;
    RangeCheck_7_2_5 range_check_7_2_5_lookup_elements;
    VerifyBitwiseXor_8 verify_bitwise_xor_8_lookup_elements;
    BlakeRound blake_round_lookup_elements;
    TripleXor32 triple_xor_32_lookup_elements;
    Opcodes opcodes_lookup_elements;
};

extern "C"
void evaluate_blake_compress_opcode(
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




#endif // EVALUATE_BLAKE_COMPRESS_OPCODE_CONSTRAINT_H