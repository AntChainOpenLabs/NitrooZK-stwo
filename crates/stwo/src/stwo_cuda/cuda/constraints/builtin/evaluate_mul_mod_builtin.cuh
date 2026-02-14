#ifndef EVALUATE_MUL_MOD_BUILTIN_H
#define EVALUATE_MUL_MOD_BUILTIN_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

struct MulModBuiltin_Claim {
    unsigned log_size;
    unsigned mul_mod_builtin_segment_start;
};

struct MulModBuiltin_Eval {
    unsigned eval_id;
    MulModBuiltin_Claim Claim;
    MemoryAddressToId memory_address_to_id_lookup_elements;
    MemoryIdToBig memory_id_to_big_lookup_elements;
    RangeCheck_12 range_check_12_lookup_elements;
    RangeCheck_3_6_6_3 range_check_3_6_6_3_lookup_elements;
    RangeCheck_18 range_check_18_lookup_elements;
};

extern "C"
void evaluate_mul_mod_builtin(
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

#endif // EVALUATE_MUL_MOD_BUILTIN_H

