#ifndef EVALUATE_DECODE_BLAKE_OPCODE_CONSTRAINT_H
#define EVALUATE_DECODE_BLAKE_OPCODE_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "constraints/evaluate_decode_instruction.cuh"
#include "constraints/evaluate_read_positive_num_bits.cuh"
#include "constraints/evaluate_read_blake_word.cuh"

DEVICE_FORCEINLINE void evaluate_decode_blake_opcode(
    // 3 input
    const m31 decode_blake_opcode_input_pc,
    const m31 decode_blake_opcode_input_ap,
    const m31 decode_blake_opcode_input_fp,
    // offsets
    const m31 offset0_col0,
    const m31 offset1_col1,
    const m31 offset2_col2,
    // base cols
    const m31 dst_base_fp_col3,
    const m31 op0_base_fp_col4,
    const m31 op1_base_fp_col5,
    const m31 ap_update_add_1_col6,
    const m31 opcode_extension_col7,
    const m31 mem0_base_col8,
    const m31 op0_id_col9,
    const m31 op0_limb_0_col10,
    const m31 op0_limb_1_col11,
    const m31 op0_limb_2_col12,
    const m31 op0_limb_3_col13,
    const m31 partial_limb_msb_col14,
    const m31 mem1_base_col15,
    const m31 op1_id_col16,
    const m31 op1_limb_0_col17,
    const m31 op1_limb_1_col18,
    const m31 op1_limb_2_col19,
    const m31 op1_limb_3_col20,
    const m31 partial_limb_msb_col21,
    const m31 ap_id_col22,
    const m31 ap_limb_0_col23,
    const m31 ap_limb_1_col24,
    const m31 ap_limb_2_col25,
    const m31 ap_limb_3_col26,
    const m31 partial_limb_msb_col27,
    const m31 mem_dst_base_col28,
    const m31 low_16_bits_col29,
    const m31 high_16_bits_col30,
    const m31 low_7_ms_bits_col31,
    const m31 high_14_ms_bits_col32,
    const m31 high_5_ms_bits_col33,
    const m31 dst_id_col34,
    // 输出
    m31* decode_blake_opcode_output_limb0,
    m31* decode_blake_opcode_output_limb1,
    m31* decode_blake_opcode_output_limb2,
    m31* decode_blake_opcode_output_limb3,
    m31* decode_blake_opcode_output_limb4,
    m31* decode_blake_opcode_output_limb5,
    m31* decode_blake_opcode_output_limb6,
    // lookup tables
    VerifyInstruction verify_instruction_lookup_elements,
    MemoryAddressToId memory_address_to_id_lookup_elements,
    MemoryIdToBig memory_id_to_big_lookup_elements,
    RangeCheck_7_2_5 range_check_7_2_5_lookup_elements,
    // evaluator
    CudaAssertEvaluator* cuda_evaluator
) {
    // 常量
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_134217728 = m31(134217728);
    const m31 M31_262144 = m31(262144);
    const m31 M31_512 = m31(512);

    // DecodeInstruction64420
    m31 decode_instruction_64420_output_tmp_47e62_10_limb[19];
    evaluate_decode_instruction_64420(
        decode_blake_opcode_input_pc,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        dst_base_fp_col3,
        op0_base_fp_col4,
        op1_base_fp_col5,
        M31_0, // op1_base_ap is always 0 for Blake (since op1_base_fp is always 1)
        ap_update_add_1_col6,
        opcode_extension_col7,
        decode_instruction_64420_output_tmp_47e62_10_limb, // [19]
        verify_instruction_lookup_elements,
        cuda_evaluator
    );

    // For Blake, op1_base_fp must be 1 (op1_base_ap doesn't exist in this component).
    cuda_evaluator->add_constraint(
        sub(op1_base_fp_col5, M31_1)
    );
    // OpcodeExtension is either Blake or BlakeFinalize.
    cuda_evaluator->add_constraint(
        mul(sub(opcode_extension_col7, M31_1), sub(opcode_extension_col7, M31_2))
    );
    // mem0_base.
    cuda_evaluator->add_constraint(
        sub(
            mem0_base_col8,
            add(
                mul(op0_base_fp_col4, decode_blake_opcode_input_fp),
                mul(sub(M31_1, op0_base_fp_col4), decode_blake_opcode_input_ap)
            )
        )
    );

    // ReadPositiveNumBits29 for op0
    m31 read_positive_num_bits_29_output_tmp_47e62_13_limb[29];
    evaluate_read_positive_num_bits_29(
        add(mem0_base_col8, decode_instruction_64420_output_tmp_47e62_10_limb[1]),
        op0_id_col9,
        op0_limb_0_col10,
        op0_limb_1_col11,
        op0_limb_2_col12,
        op0_limb_3_col13,
        partial_limb_msb_col14,
        read_positive_num_bits_29_output_tmp_47e62_13_limb, // [29]
        memory_address_to_id_lookup_elements,
        memory_id_to_big_lookup_elements,
        cuda_evaluator
    );

    // mem1_base. For Blake, op1_base_fp is always 1, so mem1_base == fp.
    cuda_evaluator->add_constraint(
        sub(
            mem1_base_col15,
            decode_blake_opcode_input_fp
        )
    );

    // ReadPositiveNumBits29 for op1
    m31 read_positive_num_bits_29_output_tmp_47e62_16_limb[29];
    evaluate_read_positive_num_bits_29(
        add(mem1_base_col15, decode_instruction_64420_output_tmp_47e62_10_limb[2]),
        op1_id_col16,
        op1_limb_0_col17,
        op1_limb_1_col18,
        op1_limb_2_col19,
        op1_limb_3_col20,
        partial_limb_msb_col21,
        read_positive_num_bits_29_output_tmp_47e62_16_limb, // [29]
        memory_address_to_id_lookup_elements,
        memory_id_to_big_lookup_elements,
        cuda_evaluator
    );

    // ReadPositiveNumBits29 for ap
    m31 read_positive_num_bits_29_output_tmp_47e62_19_limb[29];
    evaluate_read_positive_num_bits_29(
        decode_blake_opcode_input_ap,
        ap_id_col22,
        ap_limb_0_col23,
        ap_limb_1_col24,
        ap_limb_2_col25,
        ap_limb_3_col26,
        partial_limb_msb_col27,
        read_positive_num_bits_29_output_tmp_47e62_19_limb, // [29]
        memory_address_to_id_lookup_elements,
        memory_id_to_big_lookup_elements,
        cuda_evaluator
    );

    // mem_dst_base.
    cuda_evaluator->add_constraint(
        sub(
            mem_dst_base_col28,
            add(
                mul(dst_base_fp_col3, decode_blake_opcode_input_fp),
                mul(sub(M31_1, dst_base_fp_col3), decode_blake_opcode_input_ap)
            )
        )
    );

    // ReadBlakeWord for dst
    m31 read_blake_word_output_tmp_47e62_28_limb_0, read_blake_word_output_tmp_47e62_28_limb_1;
    read_blake_word_evaluate(
        add(mem_dst_base_col28, decode_instruction_64420_output_tmp_47e62_10_limb[0]),
        low_16_bits_col29,
        high_16_bits_col30,
        low_7_ms_bits_col31,
        high_14_ms_bits_col32,
        high_5_ms_bits_col33,
        dst_id_col34,
        &read_blake_word_output_tmp_47e62_28_limb_0, &read_blake_word_output_tmp_47e62_28_limb_1,
        range_check_7_2_5_lookup_elements,
        memory_address_to_id_lookup_elements,
        memory_id_to_big_lookup_elements,
        cuda_evaluator
    );

    // 输出
    *decode_blake_opcode_output_limb0 = add(add(add(op0_limb_0_col10, mul(op0_limb_1_col11, M31_512)), mul(op0_limb_2_col12, M31_262144)), mul(op0_limb_3_col13, M31_134217728));
    *decode_blake_opcode_output_limb1 = add(add(add(op1_limb_0_col17, mul(op1_limb_1_col18, M31_512)), mul(op1_limb_2_col19, M31_262144)), mul(op1_limb_3_col20, M31_134217728));
    *decode_blake_opcode_output_limb2 = add(add(add(ap_limb_0_col23, mul(ap_limb_1_col24, M31_512)), mul(ap_limb_2_col25, M31_262144)), mul(ap_limb_3_col26, M31_134217728));
    *decode_blake_opcode_output_limb3 = low_16_bits_col29;
    *decode_blake_opcode_output_limb4 = high_16_bits_col30;
    *decode_blake_opcode_output_limb5 = ap_update_add_1_col6;
    *decode_blake_opcode_output_limb6 = sub(opcode_extension_col7, M31_1);
}


#endif  // EVALUATE_DECODE_BLAKE_OPCODE_CONSTRAINT_H