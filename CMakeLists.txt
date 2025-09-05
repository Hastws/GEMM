cmake_minimum_required(VERSION 3.10)
project(bench_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(./include)

add_executable(bench_ref bench/bench_gemm_ref.cpp)

target_compile_definitions(bench_ref PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
        EIGEN_DONT_VECTORIZE)

target_compile_options(bench_ref PRIVATE
        -O3 -DNDEBUG
        -fno-tree-vectorize -fno-tree-slp-vectorize -fno-unroll-loops
        -ffp-contract=off
        -ffast-math -fno-math-errno -fno-trapping-math)


add_executable(bench_avx2 bench/bench_gemm_avx2.cpp)

target_compile_definitions(bench_avx2 PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
)

# 编译选项：开启 AVX2/FMA；允许编译器向量化与 FMA 合并
if (MSVC)
    target_compile_options(bench_avx2 PRIVATE
            /O2 /DNDEBUG
            /arch:AVX2
            /fp:fast
    )
else ()
    # 限定为 x86_64 平台时再加 AVX2 参数，其他平台给出提示但仍可编译（会走 REF 后端）
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64|X86_64")
        target_compile_options(bench_avx2 PRIVATE
                -O3 -DNDEBUG
                -mavx2 -mfma
                -ffast-math -fno-math-errno -fno-trapping-math
                # 你也可以加 -funroll-loops 视情况而定
        )
    else ()
        message(WARNING "bench_avx2: Non-x86_64 platform detected; AVX2 flags not applied. The code should fall back to REF at compile-time/headers.")
        target_compile_options(bench_avx2 PRIVATE
                -O3 -DNDEBUG
                -ffast-math -fno-math-errno -fno-trapping-math
        )
    endif ()
endif ()