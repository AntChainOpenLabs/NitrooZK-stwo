#ifndef RANGE_CHECK_19_COMMON_H
#define RANGE_CHECK_19_COMMON_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Common constraint logic for range_check_19_xxx:
// - Read Seq preprocessed column (from trace0/preprocessed)
// - Read multiplicity trace column (from trace1/base)
// - Write (-multiplicity, [seq]) to the specified RangeCheck_19_* relation
template<typename EvaluatorT, typename LookupT>
DEVICE_FORCEINLINE void eval_range_check_19_core(
    EvaluatorT &cuda_evaluator0,  // for preprocessed trace (seq)
    EvaluatorT &cuda_evaluator1,  // for base trace (multiplicity)
    LookupT lookup_elements
) {
    m31 seq_val = cuda_evaluator0.next_trace_mask();  // From trace0 (preprocessed)
    m31 multiplicity = cuda_evaluator1.next_trace_mask();  // From trace1 (base trace)

    m31 values[1] = {seq_val};

    m31 neg_mult = neg(multiplicity);
    qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

    RelationEntry<1> entry(
        lookup_elements,
        multiplicity_ext,
        values
    );
    cuda_evaluator1.add_to_relation<1>(entry);
}

#endif // RANGE_CHECK_19_COMMON_H

