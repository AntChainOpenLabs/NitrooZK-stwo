#include "evaluate_constraints.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include "timer.cuh"
#include "utils.cuh"
#include "fields.cuh"

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
) {

    CommonEval *common_eval = (CommonEval *) eval;

    if (common_eval->eval_id == 1) {
        // printf("call eval wide fib kernel\n");
        evaluate_wide_fibonacci_constraint_quotients_on_domain(
            quotients_0, quotients_1, quotients_2, quotients_3,
            trace0_evaluations,
            trace0_evaluations_len,
            trace1_evaluations,
            trace1_evaluations_len,
            random_coeff_powers,
            denominator_inverses,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns
        );
    } else if (common_eval->eval_id == 2) {
        // printf("call eval poseidon kernel\n");
        evaluate_poseidon_constraint_quotients_on_domain(
            quotients_0, quotients_1, quotients_2, quotients_3,
            trace0_evaluations,
            trace0_evaluations_len,
            trace1_evaluations,
            trace1_evaluations_len,
            trace2_evaluations,
            trace2_evaluations_len,
            random_coeff_powers,
            denominator_inverses,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            eval,
            cumsum_shift
        );
    } else {
        printf("eval id:%d not supported\n", common_eval->eval_id);
    }

}