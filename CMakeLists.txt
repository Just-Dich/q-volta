cmake_minimum_required(VERSION 3.24)
# Tesla V100 (Volta). Must be set before project(): project() otherwise pre-fills a default (52).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
option(QV_CUDA "Build the CUDA engine, kernels and tests" ON)
if(QV_CUDA)
  project(q_volta LANGUAGES CXX CUDA)
else()
  project(q_volta LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(QV_CUDA)
find_package(CUDAToolkit REQUIRED)
endif()

set(QV_CUDA_FLAGS -O3 -lineinfo -Xptxas=-v -Wno-deprecated-gpu-targets)

add_library(qv_core STATIC src/model/model_file.cpp)
target_include_directories(qv_core PUBLIC include)

if(QV_CUDA)
add_library(qv_cuda STATIC
  src/backends/cuda/kernels/gemv.cu
  src/backends/cuda/kernels/gemv_q8.cu
  src/backends/cuda/kernels/dequant.cu
  src/backends/cuda/kernels/elementwise.cu
  src/backends/cuda/kernels/linear_prefill.cu
  src/backends/cuda/kernels/elementwise_seq.cu
  src/backends/cuda/recurrent/gdn.cu
  src/backends/cuda/recurrent/gdn_seq.cu
  src/backends/cuda/attention/attn.cu
  src/backends/cuda/attention/attn_seq.cu
  src/runtime/cuda_engine.cu)
target_include_directories(qv_cuda PUBLIC include)
target_link_libraries(qv_cuda PUBLIC qv_core CUDA::cublas)
target_compile_options(qv_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${QV_CUDA_FLAGS}>)


add_executable(qv_run tools/qv_run.cpp)
target_link_libraries(qv_run PRIVATE qv_cuda CUDA::cudart)
target_compile_definitions(qv_run PRIVATE QV_HAVE_CUDA)

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
add_executable(test_gdn_seq tests/test_gdn_seq.cu)
target_link_libraries(test_gdn_seq PRIVATE qv_cuda)
target_compile_options(test_gdn_seq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME gdn_seq COMMAND test_gdn_seq)
add_executable(test_attn_decode_split tests/test_attn_decode_split.cu)
target_link_libraries(test_attn_decode_split PRIVATE qv_cuda)
target_compile_options(test_attn_decode_split PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME attn_decode_split COMMAND test_attn_decode_split)
endif()

option(QV_CPU "Build the CPU decode ops library (qv_cpu_ops)" ON)
if(QV_CPU)
  add_subdirectory(src/backends/cpu)
  if(QV_CUDA)
    target_link_libraries(qv_run PRIVATE qv_cpu_engine)
    target_compile_definitions(qv_run PRIVATE QV_HAVE_CPU)
  endif()
endif()

# ---- tokenizer / sampling / server (CPU-only capable; the CUDA engine is linked when QV_CUDA=ON) ----
add_library(qv_text STATIC src/tokenizer/tokenizer.cpp src/tokenizer/chat_template.cpp src/sampling/sampler.cpp)
target_include_directories(qv_text PUBLIC include src)
add_executable(qv_tok tools/qv_tok.cpp)
target_link_libraries(qv_tok PRIVATE qv_text)
