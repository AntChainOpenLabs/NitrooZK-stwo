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

add_library(
        stwo_cuda STATIC
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
        evaluate_poseidon_constraint.cu
        evaluate_wide_fibonacci.cu
        prefix_sum.cu
        cuda_mem_pool.cu
)
set_target_properties(stwo_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stwo_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_source_files_properties(poseidon_constraint.cu PROPERTIES
#     COMPILE_FLAGS "-Xptxas '-maxrregcount=128'"
# )

target_compile_options(stwo_cuda PRIVATE -fPIC)
target_link_libraries(stwo_cuda CUDA::curand)
target_compile_options(stwo_cuda PRIVATE --ptxas-options=-v)
install(TARGETS stwo_cuda DESTINATION .)
