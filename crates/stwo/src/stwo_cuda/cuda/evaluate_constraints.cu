#include "evaluate_constraints.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"


// Include stwo-cairo components
// Global variable to control accumulation behavior (declared extern in evaluate_common.cuh)
// This is set by the dispatcher before each component evaluation
// Note: Not thread-safe, but fine since evaluations are sequential
#include "evaluate_blake_compress_opcode.cuh"
#include "evaluate_blake_g.cuh"
#include "evaluate_triple_xor_32.cuh"
// #include "evaluate_range_check_4_3.cuh"        // Temporarily disabled
// #include "evaluate_range_check_6.cuh"           // Temporarily disabled
// #include "evaluate_range_check_7_2_5.cuh"       // Temporarily disabled
#include "evaluate_blake_round_sigma.cuh"
// #include "evaluate_memory_address_to_id.cuh"    // Temporarily disabled
// #include "evaluate_memory_id_to_big.cuh"        // Temporarily disabled
#include "evaluate_blake_round.cuh"
#include "evaluate_add_opcode.cuh"
#include "evaluate_add_ap_opcode.cuh"
#include "evaluate_add_opcode_small.cuh"
#include "evaluate_assert_eq_opcode.cuh"
#include "evaluate_assert_eq_opcode_double_deref.cuh"
#include "evaluate_assert_eq_opcode_imm.cuh"

bool g_should_accumulate_host = true;

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
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,  // whether to accumulate or overwrite
    bool use_assert_evaluator  // true for tests/debug, false for production
) {
    // Set global flag for this evaluation
    g_should_accumulate_host = should_accumulate;

    CommonEval *common_eval = (CommonEval *) eval;
    const unsigned eval_id = common_eval->eval_id;

#define DISPATCH_EVAL_SIMPLE(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        printf(MSG "\n"); \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts \
        ); \
        return; \
    }

#define DISPATCH_EVAL_WITH_BOOLS(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        printf(MSG "\n"); \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            trace2_evaluations, \
            trace2_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts, \
            eval, \
            cumsum_shift, \
            should_accumulate, \
            use_assert_evaluator \
        ); \
        return; \
    }

#define DISPATCH_EVAL_WITH_CUMSUM(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        printf(MSG "\n"); \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            trace2_evaluations, \
            trace2_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts, \
            eval, \
            cumsum_shift \
        ); \
        return; \
    }

    switch (eval_id) {
        DISPATCH_EVAL_SIMPLE("fibonacci_example", "call cuda eval wide fib example", evaluate_wide_fibonacci_constraint_quotients_on_domain);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_example", "call cuda eval poseidon example", evaluate_poseidon_constraint_quotients_on_domain);

        // stwo-cairo components
        DISPATCH_EVAL_WITH_BOOLS("blake_compress_opcode", "call cuda eval stwo-cairo blake_compress_opcode", evaluate_blake_compress_opcode);
        DISPATCH_EVAL_WITH_BOOLS("blake_g", "call cuda eval stwo-cairo blake_g", evaluate_blake_g);
        DISPATCH_EVAL_WITH_BOOLS("triple_xor_32", "call cuda eval stwo-cairo triple_xor_32", evaluate_triple_xor_32);
        // DISPATCH_EVAL_WITH_BOOLS("range_check_4_3", "call cuda eval stwo-cairo range_check_4_3", evaluate_range_check_4_3);  // Disabled
        // DISPATCH_EVAL_WITH_BOOLS("range_check_6", "call cuda eval stwo-cairo range_check_6", evaluate_range_check_6);  // Disabled
        // DISPATCH_EVAL_WITH_BOOLS("range_check_7_2_5", "call cuda eval stwo-cairo range_check_7_2_5", evaluate_range_check_7_2_5);  // Disabled
        DISPATCH_EVAL_WITH_BOOLS("blake_round_sigma", "call cuda eval stwo-cairo blake_round_sigma", evaluate_blake_round_sigma);
        // DISPATCH_EVAL_WITH_BOOLS("memory_address_to_id", "call cuda eval stwo-cairo memory_address_to_id", evaluate_memory_address_to_id);  // Disabled
        // DISPATCH_EVAL_WITH_BOOLS("memory_id_to_big", "call cuda eval stwo-cairo memory_id_to_big", evaluate_memory_id_to_big);  // Disabled
        DISPATCH_EVAL_WITH_BOOLS("blake_round", "call cuda eval stwo-cairo blake_round", evaluate_blake_round);
        DISPATCH_EVAL_WITH_BOOLS("add_opcode", "call cuda eval stwo-cairo add_opcode", evaluate_add_opcode);
        DISPATCH_EVAL_WITH_BOOLS("add_ap_opcode", "call cuda eval stwo-cairo add_ap_opcode", evaluate_add_ap_opcode);
        DISPATCH_EVAL_WITH_BOOLS("add_opcode_small", "call cuda eval stwo-cairo add_opcode_small", evaluate_add_opcode_small);
        DISPATCH_EVAL_WITH_BOOLS("assert_eq_opcode", "call cuda eval stwo-cairo assert_eq_opcode", evaluate_assert_eq_opcode);
        DISPATCH_EVAL_WITH_BOOLS("assert_eq_opcode_double_deref", "call cuda eval stwo-cairo assert_eq_opcode_double_deref", evaluate_assert_eq_opcode_double_deref);
        DISPATCH_EVAL_WITH_CUMSUM("assert_eq_opcode_imm", "call cuda eval stwo-cairo assert_eq_opcode_imm", evaluate_assert_eq_opcode_imm);

        default:
            printf("eval id:%u not supported\n", eval_id);
    }

#undef DISPATCH_EVAL_SIMPLE
#undef DISPATCH_EVAL_WITH_BOOLS
#undef DISPATCH_EVAL_WITH_CUMSUM
}
