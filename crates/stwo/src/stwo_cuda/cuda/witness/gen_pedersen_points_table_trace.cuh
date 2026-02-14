// CUDA interaction trace generation for pedersen_points_table component.
//
// The pedersen_points_table is a preprocessed lookup table (~8M rows, LOG_SIZE=23)
// with 56 columns of EC point coordinates. The interaction trace has a single logup
// column: numerator = -mults[row], denominator = lookup_elements.combine(57 values)
// where values = [seq (=row_index), table_col_0..table_col_55].

#ifndef GEN_PEDERSEN_POINTS_TABLE_TRACE_CUH
#define GEN_PEDERSEN_POINTS_TABLE_TRACE_CUH

#include "fields.cuh"

// Number of lookup values per row: 1 (seq) + 56 (table columns)
#define PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES 57

extern "C"
void pedersen_points_table_interaction_trace(
    void *lookup_elements,          // LookupElementsBasic<57> from Rust
    m31 *multiplicities,            // GPU multiplicities (u32 per row)
    unsigned log_size,              // Log2 of table size (23)
    m31 **interaction_traces,       // 4 output columns (qm31 components)
    m31 *claimed_sum                // Output claimed sum (4 x m31)
);

#endif // GEN_PEDERSEN_POINTS_TABLE_TRACE_CUH
