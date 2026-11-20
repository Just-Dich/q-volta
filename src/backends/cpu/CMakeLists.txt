# Q-Volta CPU ops. Usable standalone (cmake -S src/backends/cpu -B build; needs only a C++20 compiler) or through
# add_subdirectory from the top-level project (option QV_CPU).
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(qv_cpu_ops LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  enable_testing()
endif()
find_package(Threads REQUIRED)
set(QV_CPU_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)

add_library(qv_cpu_ops STATIC ${CMAKE_CURRENT_LIST_DIR}/ops.cpp)
target_include_directories(qv_cpu_ops PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(qv_cpu_ops PRIVATE -O3 -mavx2 -mfma -mf16c)

add_executable(test_cpu_ops ${QV_CPU_ROOT}/tests/test_cpu_ops.cpp)
target_link_libraries(test_cpu_ops PRIVATE qv_cpu_ops)
target_compile_options(test_cpu_ops PRIVATE -O2 -mavx2 -mfma -mf16c)
add_test(NAME cpu_ops COMMAND test_cpu_ops)

add_executable(bench_cpu_ops ${QV_CPU_ROOT}/benchmarks/bench_cpu_ops.cpp)
target_link_libraries(bench_cpu_ops PRIVATE qv_cpu_ops Threads::Threads)
target_compile_options(bench_cpu_ops PRIVATE -O3 -mavx2 -mfma -mf16c)
