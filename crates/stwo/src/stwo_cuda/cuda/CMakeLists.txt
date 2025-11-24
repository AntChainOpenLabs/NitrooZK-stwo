cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

project(stwo_cuda LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr ")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUH_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cuh")
set(CUH_DIRS "")
foreach(file_path ${CUH_FILES})
    get_filename_component(dir ${file_path} DIRECTORY)
    list(APPEND CUH_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES CUH_DIRS)

message(STATUS "Found cuh directories: ${CUH_DIRS}")

add_library(
        stwo_cuda STATIC
        # Core CUDA utilities
        batch_inverse.cu
        utils.cu
        bit_reverse.cu
        eval_at_point.cu
        ifft.cu
        rfft.cu
        twiddles.cu
        poly_utils.cu
        fold_line.cu
        fold_circle_into_line.cu
        fri_utils.cu
        accumulate.cu
        fields.cu
        blake2s.cu
        poseidon252_starknet.cu
        quotients.cu
        gkr.cu
        mle.cu
        timer.cu
        evaluate_constraints.cu
        evaluate_common.cu
        evaluate_poseidon_constraint.cu
        evaluate_wide_fibonacci.cu
        prefix_sum.cu
        cuda_mem_pool.cu
        # stwo-cairo AIR components
        constraints/evaluate_blake_compress_opcode.cu
        constraints/evaluate_blake_g.cu
        constraints/evaluate_triple_xor_32.cu
        constraints/evaluate_blake_round_sigma.cu
        constraints/evaluate_blake_round.cu
        constraints/evaluate_add_opcode.cu
        constraints/evaluate_add_ap_opcode.cu
        constraints/evaluate_add_opcode_small.cu
        constraints/evaluate_assert_eq_opcode.cu
        constraints/evaluate_assert_eq_opcode_double_deref.cu
        constraints/evaluate_assert_eq_opcode_imm.cu
        witness/gen_blake_g_trace.cu
        witness/gen_add_opcode_trace.cu
        witness/gen_add_opcode_small_trace.cu
        witness/gen_add_ap_opcode_trace.cu
        witness/gen_assert_eq_opcode_trace.cu
        witness/gen_assert_eq_opcode_double_deref_trace.cu
        witness/gen_assert_eq_opcode_imm_trace.cu
        witness/gen_triple_xor_32_trace.cu
        witness/gen_verify_bitwise_xor_trace.cu
        witness/gen_blake_round_trace.cu
        witness/gen_memory_address_to_id_trace.cu
        witness/gen_blake_round_sigma_trace.cu
        witness/gen_memory_id_to_big_trace.cu
)
set_target_properties(stwo_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stwo_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(stwo_cuda PRIVATE -fPIC)
target_link_libraries(stwo_cuda CUDA::curand)
target_compile_options(stwo_cuda PRIVATE --ptxas-options=-v)
target_link_libraries(stwo_cuda CUDA::curand)
target_include_directories(stwo_cuda PRIVATE ${CUH_DIRS})
install(TARGETS stwo_cuda DESTINATION .)
