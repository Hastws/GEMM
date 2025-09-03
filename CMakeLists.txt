cmake_minimum_required(VERSION 3.10)
project(bench_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen3 include dir")

#option(AUTOALG_ENABLE_AVX2 "Build AVX2 backend (with runtime fallback)" ON)

include_directories(./include)

add_executable(bench bench/bench_gemm.cpp)

#if (AUTOALG_ENABLE_AVX2)
#    add_library(autoalg_avx2 INTERFACE)
#    if (MSVC)
#        target_compile_options(autoalg_avx2 INTERFACE /arch:AVX2)
#    else()
#        target_compile_options(autoalg_avx2 INTERFACE -mavx2 -mfma)
#    endif()
#    target_compile_definitions(autoalg_avx2 INTERFACE AUTOALG_HAVE_AVX2=1)
#    target_link_libraries(bench PRIVATE autoalg_avx2)
#endif()

target_include_directories(bench PRIVATE ${EIGEN3_INCLUDE_DIR} .)

target_compile_definitions(bench PRIVATE EIGEN_NO_DEBUG)
target_compile_options(bench PRIVATE -O3 -DNDEBUG)
