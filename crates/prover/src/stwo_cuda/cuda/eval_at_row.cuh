#ifndef EVAL_AT_ROW_H
#define EVAL_AT_ROW_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"

struct Fraction {
    qm31 numerator;
    qm31 denominator;

    HOST_DEVICE_FORCEINLINE Fraction() : numerator(qm31{{0, 0}, {0, 0}}), denominator(qm31{{1, 0}, {0, 0}}) {}

    HOST_DEVICE_FORCEINLINE Fraction(const qm31 numerator, const qm31 denominator)
        : numerator(numerator), denominator(denominator) {}

    HOST_DEVICE_FORCEINLINE static Fraction zero() {
        return Fraction(qm31{{0, 0}, {0, 0}}, qm31{{1, 0}, {0, 0}});
    }

    HOST_DEVICE_FORCEINLINE bool is_zero() const {
        qm31 zero = {{0, 0}, {0, 0}};
        qm31 one = {{1, 0}, {0, 0}};

        bool numerator_is_zero =
            (numerator.a.a == zero.a.a) &&
            (numerator.a.b == zero.a.b) &&
            (numerator.b.a == zero.b.a) &&
            (numerator.b.b == zero.b.b);

        bool denominator_is_one =
            (denominator.a.a == one.a.a) &&
            (denominator.a.b == one.a.b) &&
            (denominator.b.a == one.b.a) &&
            (denominator.b.b == one.b.b);

        return numerator_is_zero && denominator_is_one;
    }

    HOST_DEVICE_FORCEINLINE void dump(const char *description) {
        printf("%s, Fraction: {numerator: (%d + %di) + (%d + %di)u, denominator: (%d + %di) + (%d + %di)u}\n", description, numerator.a.a, numerator.a.b, numerator.b.a, numerator.b.b, denominator.a.a, denominator.a.b, denominator.b.a, denominator.b.b);
    }

    // sum to first element
    HOST_DEVICE_FORCEINLINE static Fraction sum(Fraction *fractions, unsigned len) {
        Fraction *sum = &fractions[0];
        for (unsigned i = 1; i < len; i++) {
            sum->numerator = add(mul(fractions[i].numerator, sum->denominator), mul(fractions[i].denominator, sum->numerator));
            sum->denominator = mul(fractions[i].denominator, sum->denominator);
        }
        return *sum;
    }
};

template <int N>
struct RelationEntry{
    LookupElementsBasic<N> relation;
    qm31 multiplicity;
    m31 values[N];

    HOST_DEVICE_FORCEINLINE RelationEntry(LookupElementsBasic<N> relation, qm31 multiplicity, m31* values)
    : relation(relation), multiplicity(multiplicity) {
        for (int i = 0; i < N; ++i) {
            this->values[i] = values[i];
        }
    }
};

DEVICE_FORCEINLINE void next_interaction_mask(
    m31 **trace_evaluations,
    unsigned *col_index,
    unsigned interaction,
    const int *offsets,
    unsigned N,
    unsigned row,
    unsigned domain_log_size,
    unsigned eval_domain_log_size,
    m31 *result
) {
    unsigned current_col_index = col_index[interaction];
    col_index[interaction] += 1;

    for (unsigned i = 0; i < N; ++i) {
        int off = offsets[i];
        if (off == 0) {
            result[i] = trace_evaluations[current_col_index][row];
        } else {
            int target_row = offset_bit_reversed_circle_domain_index(
                row, domain_log_size, eval_domain_log_size, off
            );
            result[i] = trace_evaluations[current_col_index][target_row];
        }
    }
}

// next_trace_mask
DEVICE_FORCEINLINE m31 next_trace_mask(
    m31 **trace_evaluations,
    unsigned *col_index,
    unsigned row,
    unsigned domain_log_size,
    unsigned eval_domain_log_size
) {
    m31 result[1];
    int offsets[1] = {0};
    next_interaction_mask(
        trace_evaluations, col_index, 0, offsets, 1,
        row, domain_log_size, eval_domain_log_size, result
    );
    return result[0];
}

DEVICE_FORCEINLINE void add_constraint(
    qm31 *random_coeff_powers,
    unsigned *constraint_index,
    m31 constraint,
    qm31 *row_res
) {
    *row_res = add(*row_res, mul(constraint, random_coeff_powers[*constraint_index]));
    (*constraint_index)++;
}

template<int N>
DEVICE_FORCEINLINE Fraction add_to_relation(
    RelationEntry<N> entry
) {
    Fraction fraction = Fraction(entry.multiplicity, entry.relation.combine(entry.values, N));
    return fraction;
}

HOST_DEVICE_FORCEINLINE void add_constraint_ext(
    qm31 *random_coeff_powers,
    unsigned *constraint_index,
    qm31 constraint,
    qm31 *row_res
) {
    *row_res = add(*row_res, mul(constraint, random_coeff_powers[*constraint_index]));
    (*constraint_index)++;
}


DEVICE_FORCEINLINE qm31 combine_ef(m31 values[SECURE_EXTENSION_DEGREE]) {
    qm31 result;
    result.a.a = values[0];
    result.a.b = values[1];
    result.b.a = values[2];
    result.b.b = values[3];

    return result;
}


DEVICE_FORCEINLINE void next_extension_interaction_mask(
    m31 **trace_evaluations,
    unsigned *col_index,
    unsigned interaction,
    const int *offsets,
    unsigned N,
    unsigned row,
    unsigned domain_log_size,
    unsigned eval_domain_log_size,
    qm31 *result
) {
    const unsigned N_MAX = 4;
    m31 base_results[SECURE_EXTENSION_DEGREE][N_MAX];
    for (unsigned i = 0; i < SECURE_EXTENSION_DEGREE; ++i) {
        next_interaction_mask(
            trace_evaluations, col_index, interaction, offsets, N,
            row, domain_log_size, eval_domain_log_size, base_results[i]
        );
    }

    for (unsigned i = 0; i < N; ++i) {
        m31 values[SECURE_EXTENSION_DEGREE];
        for (unsigned j = 0; j < SECURE_EXTENSION_DEGREE; ++j) {
            values[j] = base_results[j][i];
        }
        result[i] = combine_ef(values);
    }
}

#endif