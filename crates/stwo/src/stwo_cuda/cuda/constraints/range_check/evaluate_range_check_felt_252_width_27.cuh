#ifndef EVALUATE_RANGE_CHECK_FELT_252_WIDTH_27_H
#define EVALUATE_RANGE_CHECK_FELT_252_WIDTH_27_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

struct RangeCheckFelt252Width27_Claim {
    unsigned log_size;
};

struct RangeCheckFelt252Width27_Eval {
    unsigned eval_id;
    RangeCheckFelt252Width27_Claim Claim;
    RangeCheck_9_9 range_check_9_9_lookup_elements;
    RangeCheck_18 range_check_18_lookup_elements;
    RangeCheck_9_9_B range_check_9_9_b_lookup_elements;
    RangeCheck_18_B range_check_18_b_lookup_elements;
    RangeCheck_9_9_C range_check_9_9_c_lookup_elements;
    RangeCheck_9_9_D range_check_9_9_d_lookup_elements;
    RangeCheck_9_9_E range_check_9_9_e_lookup_elements;
    RangeCheckFelt252Width27 range_check_felt_252_width_27_lookup_elements;
};

extern "C"
void evaluate_range_check_felt_252_width_27(
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

#endif // EVALUATE_RANGE_CHECK_FELT_252_WIDTH_27_H

