cmake_minimum_required(VERSION 3.24)
# Tesla V100 (Volta). Must be set before project(): project() otherwise pre-fills a default (52).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
project(q_volta LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)

set(QV_CUDA_FLAGS -O3 -lineinfo -Xptxas=-v -Wno-deprecated-gpu-targets)

add_library(qv_core STATIC src/model/model_file.cpp)
target_include_directories(qv_core PUBLIC include)

add_library(qv_cuda STATIC
  src/backends/cuda/kernels/gemv.cu
  src/backends/cuda/kernels/gemv_q8.cu
  src/backends/cuda/kernels/dequant.cu
  src/backends/cuda/kernels/elementwise.cu
  src/backends/cuda/kernels/linear_prefill.cu
  src/backends/cuda/kernels/elementwise_seq.cu
  src/backends/cuda/recurrent/gdn.cu
  src/backends/cuda/attention/attn.cu
  src/backends/cuda/attention/attn_seq.cu
  src/runtime/cuda_engine.cu)
target_include_directories(qv_cuda PUBLIC include)
target_link_libraries(qv_cuda PUBLIC qv_core CUDA::cublas)
target_compile_options(qv_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${QV_CUDA_FLAGS}>)

add_executable(qv_run tools/qv_run.cpp)
target_link_libraries(qv_run PRIVATE qv_cuda CUDA::cudart)

enable_testing()
add_executable(test_gemv tests/test_gemv.cu)
target_link_libraries(test_gemv PRIVATE qv_cuda)
target_compile_options(test_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME gemv COMMAND test_gemv ${CMAKE_BINARY_DIR}/gemv_cases)
add_executable(bench_gemv benchmarks/bench_gemv.cu)
target_link_libraries(bench_gemv PRIVATE qv_cuda)
target_compile_options(bench_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_executable(test_gemv_multi tests/test_gemv_multi.cu)
target_link_libraries(test_gemv_multi PRIVATE qv_cuda)
target_compile_options(test_gemv_multi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME gemv_multi COMMAND test_gemv_multi)
add_executable(test_dequant tests/test_dequant.cu)
target_link_libraries(test_dequant PRIVATE qv_cuda)
target_compile_options(test_dequant PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME dequant COMMAND test_dequant)
add_executable(test_attn_seq tests/test_attn_seq.cu)
target_link_libraries(test_attn_seq PRIVATE qv_cuda)
target_compile_options(test_attn_seq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME attn_seq COMMAND test_attn_seq)
