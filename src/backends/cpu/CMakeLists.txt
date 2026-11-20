# CPU backend (AVX2). Standalone-capable: `cmake -S src/backends/cpu -B build-cpu` needs only a C++20 compiler
# (no CUDA). Also pulled into the main build via add_subdirectory.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(q_volta_cpu LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  enable_testing()
endif()
get_filename_component(QV_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(QV_CPU_FLAGS -O3 -mavx2 -mfma -mf16c -mbmi2)

add_library(qv_cpu STATIC cpu_gemv.cpp)
target_include_directories(qv_cpu PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(qv_cpu PUBLIC ${QV_CPU_FLAGS})
target_link_libraries(qv_cpu PUBLIC pthread)

add_executable(test_cpu_gemv ${QV_ROOT}/tests/test_cpu_gemv.cpp)
target_link_libraries(test_cpu_gemv PRIVATE qv_cpu)
add_test(NAME cpu_gemv COMMAND test_cpu_gemv ${QV_ROOT}/tests/data/pack_fixture.txt)

add_executable(bench_cpu_gemv ${QV_ROOT}/benchmarks/bench_cpu_gemv.cpp)
target_link_libraries(bench_cpu_gemv PRIVATE qv_cpu)
