#ifndef EVALUATE_CONSTRAINT_H
#define EVALUATE_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"

struct CommonEval {
    unsigned eval_id;
    unsigned log_n_rows;
};

extern "C"
void evaluate_constraint_quotients_on_domain_new(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *constraints_vec,
    m31 *random_coeff_powers_vec,
    unsigned int constraints_coeff_pair_col_num,
    unsigned int constraints_coeff_pair_row_num,
    unsigned int trace_domain_log_size
);

extern "C"
void evaluate_constraint_quotients_on_domain_new_simd(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 *denominator_inverses,
    m31 *simd_constraints_vec,
    m31 *simd_random_coeff_powers_vec,
    unsigned int constraints_num,
    unsigned int vec_rows_num,
    unsigned int trace_domain_log_size
);

extern "C"
void evaluate_constraint_quotients_on_domain(
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
    void *eval,
    qm31 cumsum_shift
);

#endif