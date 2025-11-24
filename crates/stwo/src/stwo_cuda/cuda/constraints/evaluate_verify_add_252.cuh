#ifndef EVALUATE_VERIFY_ADD_252_H
#define EVALUATE_VERIFY_ADD_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_verify_add_252(
    const m31 verify_add_252_input_limb_0,
    const m31 verify_add_252_input_limb_1,
    const m31 verify_add_252_input_limb_2,
    const m31 verify_add_252_input_limb_3,
    const m31 verify_add_252_input_limb_4,
    const m31 verify_add_252_input_limb_5,
    const m31 verify_add_252_input_limb_6,
    const m31 verify_add_252_input_limb_7,
    const m31 verify_add_252_input_limb_8,
    const m31 verify_add_252_input_limb_9,
    const m31 verify_add_252_input_limb_10,
    const m31 verify_add_252_input_limb_11,
    const m31 verify_add_252_input_limb_12,
    const m31 verify_add_252_input_limb_13,
    const m31 verify_add_252_input_limb_14,
    const m31 verify_add_252_input_limb_15,
    const m31 verify_add_252_input_limb_16,
    const m31 verify_add_252_input_limb_17,
    const m31 verify_add_252_input_limb_18,
    const m31 verify_add_252_input_limb_19,
    const m31 verify_add_252_input_limb_20,
    const m31 verify_add_252_input_limb_21,
    const m31 verify_add_252_input_limb_22,
    const m31 verify_add_252_input_limb_23,
    const m31 verify_add_252_input_limb_24,
    const m31 verify_add_252_input_limb_25,
    const m31 verify_add_252_input_limb_26,
    const m31 verify_add_252_input_limb_27,
    const m31 verify_add_252_input_limb_28,
    const m31 verify_add_252_input_limb_29,
    const m31 verify_add_252_input_limb_30,
    const m31 verify_add_252_input_limb_31,
    const m31 verify_add_252_input_limb_32,
    const m31 verify_add_252_input_limb_33,
    const m31 verify_add_252_input_limb_34,
    const m31 verify_add_252_input_limb_35,
    const m31 verify_add_252_input_limb_36,
    const m31 verify_add_252_input_limb_37,
    const m31 verify_add_252_input_limb_38,
    const m31 verify_add_252_input_limb_39,
    const m31 verify_add_252_input_limb_40,
    const m31 verify_add_252_input_limb_41,
    const m31 verify_add_252_input_limb_42,
    const m31 verify_add_252_input_limb_43,
    const m31 verify_add_252_input_limb_44,
    const m31 verify_add_252_input_limb_45,
    const m31 verify_add_252_input_limb_46,
    const m31 verify_add_252_input_limb_47,
    const m31 verify_add_252_input_limb_48,
    const m31 verify_add_252_input_limb_49,
    const m31 verify_add_252_input_limb_50,
    const m31 verify_add_252_input_limb_51,
    const m31 verify_add_252_input_limb_52,
    const m31 verify_add_252_input_limb_53,
    const m31 verify_add_252_input_limb_54,
    const m31 verify_add_252_input_limb_55,
    const m31 verify_add_252_input_limb_56,
    const m31 verify_add_252_input_limb_57,
    const m31 verify_add_252_input_limb_58,
    const m31 verify_add_252_input_limb_59,
    const m31 verify_add_252_input_limb_60,
    const m31 verify_add_252_input_limb_61,
    const m31 verify_add_252_input_limb_62,
    const m31 verify_add_252_input_limb_63,
    const m31 verify_add_252_input_limb_64,
    const m31 verify_add_252_input_limb_65,
    const m31 verify_add_252_input_limb_66,
    const m31 verify_add_252_input_limb_67,
    const m31 verify_add_252_input_limb_68,
    const m31 verify_add_252_input_limb_69,
    const m31 verify_add_252_input_limb_70,
    const m31 verify_add_252_input_limb_71,
    const m31 verify_add_252_input_limb_72,
    const m31 verify_add_252_input_limb_73,
    const m31 verify_add_252_input_limb_74,
    const m31 verify_add_252_input_limb_75,
    const m31 verify_add_252_input_limb_76,
    const m31 verify_add_252_input_limb_77,
    const m31 verify_add_252_input_limb_78,
    const m31 verify_add_252_input_limb_79,
    const m31 verify_add_252_input_limb_80,
    const m31 verify_add_252_input_limb_81,
    const m31 verify_add_252_input_limb_82,
    const m31 verify_add_252_input_limb_83,
    const m31 sub_p_bit_col0,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_1 = m31(1);
    m31 M31_136 = m31(136);
    m31 M31_256 = m31(256);
    m31 M31_4194304 = m31(4194304);

    cuda_evaluator->add_constraint(mul(sub_p_bit_col0, sub(sub_p_bit_col0, M31_1)));

    m31 carry_tmp_4afb1_1 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(verify_add_252_input_limb_0, verify_add_252_input_limb_28),
                add(verify_add_252_input_limb_56, sub_p_bit_col0)
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_1,
            sub(mul(carry_tmp_4afb1_1, carry_tmp_4afb1_1), M31_1)
        )
    );

    m31 carry_tmp_4afb1_2 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_1, verify_add_252_input_limb_29), carry_tmp_4afb1_1),
                verify_add_252_input_limb_57
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_2,
            sub(mul(carry_tmp_4afb1_2, carry_tmp_4afb1_2), M31_1)
        )
    );

    m31 carry_tmp_4afb1_3 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_2, verify_add_252_input_limb_30), carry_tmp_4afb1_2),
                verify_add_252_input_limb_58
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_3,
            sub(mul(carry_tmp_4afb1_3, carry_tmp_4afb1_3), M31_1)
        )
    );

    m31 carry_tmp_4afb1_4 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_3, verify_add_252_input_limb_31), carry_tmp_4afb1_3),
                verify_add_252_input_limb_59
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_4,
            sub(mul(carry_tmp_4afb1_4, carry_tmp_4afb1_4), M31_1)
        )
    );

    m31 carry_tmp_4afb1_5 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_4, verify_add_252_input_limb_32), carry_tmp_4afb1_4),
                verify_add_252_input_limb_60
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_5,
            sub(mul(carry_tmp_4afb1_5, carry_tmp_4afb1_5), M31_1)
        )
    );

    m31 carry_tmp_4afb1_6 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_5, verify_add_252_input_limb_33), carry_tmp_4afb1_5),
                verify_add_252_input_limb_61
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_6,
            sub(mul(carry_tmp_4afb1_6, carry_tmp_4afb1_6), M31_1)
        )
    );

    m31 carry_tmp_4afb1_7 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_6, verify_add_252_input_limb_34), carry_tmp_4afb1_6),
                verify_add_252_input_limb_62
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_7,
            sub(mul(carry_tmp_4afb1_7, carry_tmp_4afb1_7), M31_1)
        )
    );

    m31 carry_tmp_4afb1_8 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_7, verify_add_252_input_limb_35), carry_tmp_4afb1_7),
                verify_add_252_input_limb_63
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_8,
            sub(mul(carry_tmp_4afb1_8, carry_tmp_4afb1_8), M31_1)
        )
    );

    m31 carry_tmp_4afb1_9 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_8, verify_add_252_input_limb_36), carry_tmp_4afb1_8),
                verify_add_252_input_limb_64
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_9,
            sub(mul(carry_tmp_4afb1_9, carry_tmp_4afb1_9), M31_1)
        )
    );

    m31 carry_tmp_4afb1_10 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_9, verify_add_252_input_limb_37), carry_tmp_4afb1_9),
                verify_add_252_input_limb_65
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_10,
            sub(mul(carry_tmp_4afb1_10, carry_tmp_4afb1_10), M31_1)
        )
    );

    m31 carry_tmp_4afb1_11 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_10, verify_add_252_input_limb_38), carry_tmp_4afb1_10),
                verify_add_252_input_limb_66
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_11,
            sub(mul(carry_tmp_4afb1_11, carry_tmp_4afb1_11), M31_1)
        )
    );

    m31 carry_tmp_4afb1_12 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_11, verify_add_252_input_limb_39), carry_tmp_4afb1_11),
                verify_add_252_input_limb_67
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_12,
            sub(mul(carry_tmp_4afb1_12, carry_tmp_4afb1_12), M31_1)
        )
    );

    m31 carry_tmp_4afb1_13 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_12, verify_add_252_input_limb_40), carry_tmp_4afb1_12),
                verify_add_252_input_limb_68
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_13,
            sub(mul(carry_tmp_4afb1_13, carry_tmp_4afb1_13), M31_1)
        )
    );

    m31 carry_tmp_4afb1_14 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_13, verify_add_252_input_limb_41), carry_tmp_4afb1_13),
                verify_add_252_input_limb_69
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_14,
            sub(mul(carry_tmp_4afb1_14, carry_tmp_4afb1_14), M31_1)
        )
    );

    m31 carry_tmp_4afb1_15 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_14, verify_add_252_input_limb_42), carry_tmp_4afb1_14),
                verify_add_252_input_limb_70
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_15,
            sub(mul(carry_tmp_4afb1_15, carry_tmp_4afb1_15), M31_1)
        )
    );

    m31 carry_tmp_4afb1_16 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_15, verify_add_252_input_limb_43), carry_tmp_4afb1_15),
                verify_add_252_input_limb_71
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_16,
            sub(mul(carry_tmp_4afb1_16, carry_tmp_4afb1_16), M31_1)
        )
    );

    m31 carry_tmp_4afb1_17 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_16, verify_add_252_input_limb_44), carry_tmp_4afb1_16),
                verify_add_252_input_limb_72
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_17,
            sub(mul(carry_tmp_4afb1_17, carry_tmp_4afb1_17), M31_1)
        )
    );

    m31 carry_tmp_4afb1_18 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_17, verify_add_252_input_limb_45), carry_tmp_4afb1_17),
                verify_add_252_input_limb_73
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_18,
            sub(mul(carry_tmp_4afb1_18, carry_tmp_4afb1_18), M31_1)
        )
    );

    m31 carry_tmp_4afb1_19 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_18, verify_add_252_input_limb_46), carry_tmp_4afb1_18),
                verify_add_252_input_limb_74
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_19,
            sub(mul(carry_tmp_4afb1_19, carry_tmp_4afb1_19), M31_1)
        )
    );

    m31 carry_tmp_4afb1_20 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_19, verify_add_252_input_limb_47), carry_tmp_4afb1_19),
                verify_add_252_input_limb_75
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_20,
            sub(mul(carry_tmp_4afb1_20, carry_tmp_4afb1_20), M31_1)
        )
    );

    m31 carry_tmp_4afb1_21 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_20, verify_add_252_input_limb_48), carry_tmp_4afb1_20),
                verify_add_252_input_limb_76
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_21,
            sub(mul(carry_tmp_4afb1_21, carry_tmp_4afb1_21), M31_1)
        )
    );

    m31 carry_tmp_4afb1_22 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                sub(
                    add(add(verify_add_252_input_limb_21, verify_add_252_input_limb_49), carry_tmp_4afb1_21),
                    verify_add_252_input_limb_77
                ),
                mul(M31_136, sub_p_bit_col0)
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_22,
            sub(mul(carry_tmp_4afb1_22, carry_tmp_4afb1_22), M31_1)
        )
    );

    m31 carry_tmp_4afb1_23 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_22, verify_add_252_input_limb_50), carry_tmp_4afb1_22),
                verify_add_252_input_limb_78
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_23,
            sub(mul(carry_tmp_4afb1_23, carry_tmp_4afb1_23), M31_1)
        )
    );

    m31 carry_tmp_4afb1_24 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_23, verify_add_252_input_limb_51), carry_tmp_4afb1_23),
                verify_add_252_input_limb_79
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_24,
            sub(mul(carry_tmp_4afb1_24, carry_tmp_4afb1_24), M31_1)
        )
    );

    m31 carry_tmp_4afb1_25 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_24, verify_add_252_input_limb_52), carry_tmp_4afb1_24),
                verify_add_252_input_limb_80
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_25,
            sub(mul(carry_tmp_4afb1_25, carry_tmp_4afb1_25), M31_1)
        )
    );

    m31 carry_tmp_4afb1_26 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_25, verify_add_252_input_limb_53), carry_tmp_4afb1_25),
                verify_add_252_input_limb_81
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_26,
            sub(mul(carry_tmp_4afb1_26, carry_tmp_4afb1_26), M31_1)
        )
    );

    m31 carry_tmp_4afb1_27 = cuda_evaluator->add_intermediate(
        mul(
            sub(
                add(add(verify_add_252_input_limb_26, verify_add_252_input_limb_54), carry_tmp_4afb1_26),
                verify_add_252_input_limb_82
            ),
            M31_4194304
        )
    );
    cuda_evaluator->add_constraint(
        mul(
            carry_tmp_4afb1_27,
            sub(mul(carry_tmp_4afb1_27, carry_tmp_4afb1_27), M31_1)
        )
    );

    cuda_evaluator->add_constraint(
        sub(
            sub(
                add(add(verify_add_252_input_limb_27, verify_add_252_input_limb_55), carry_tmp_4afb1_27),
                verify_add_252_input_limb_83
            ),
            mul(M31_256, sub_p_bit_col0)
        )
    );
}


#endif // EVALUATE_VERIFY_ADD_252_H