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

# ============ NEON Backend (ARM) ============
add_executable(bench_neon bench/bench_gemm_neon.cpp)

target_compile_definitions(bench_neon PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
)

# 限定为 ARM/AArch64 平台时加 NEON 参数，其他平台给出提示但仍可编译（会走 REF 后端）
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64|armv7|armv8")
    target_compile_options(bench_neon PRIVATE
            -O3 -DNDEBUG
            -ffast-math -fno-math-errno -fno-trapping-math
    )
    # AArch64 默认启用 NEON，无需额外标志；ARMv7 可能需要 -mfpu=neon
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        target_compile_options(bench_neon PRIVATE -mfpu=neon)
    endif ()
else ()
    message(WARNING "bench_neon: Non-ARM platform detected; NEON flags not applied. The code should fall back to REF at compile-time/headers.")
    target_compile_options(bench_neon PRIVATE
            -O3 -DNDEBUG
            -ffast-math -fno-math-errno -fno-trapping-math
    )
endif ()

# ============ Blocked (Cache Tiling) Backend ============
add_executable(bench_blocked bench/bench_gemm_blocked.cpp)

target_compile_definitions(bench_blocked PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
)

target_compile_options(bench_blocked PRIVATE
        -O3 -DNDEBUG
        -ffast-math -fno-math-errno -fno-trapping-math
)

# ============ OpenMP Backend ============
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(bench_openmp bench/bench_gemm_openmp.cpp)
    
    target_compile_definitions(bench_openmp PRIVATE
            EIGEN_NO_DEBUG
    )
    
    target_compile_options(bench_openmp PRIVATE
            -O3 -DNDEBUG
            -ffast-math -fno-math-errno -fno-trapping-math
    )
    
    target_link_libraries(bench_openmp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found, bench_openmp will not be built")
endif()

# ============ AVX2 + Blocked Backend ============
add_executable(bench_avx2_blocked bench/bench_gemm_avx2_blocked.cpp)

target_compile_definitions(bench_avx2_blocked PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64|X86_64")
    target_compile_options(bench_avx2_blocked PRIVATE
            -O3 -DNDEBUG
            -mavx2 -mfma
            -ffast-math -fno-math-errno -fno-trapping-math
    )
else()
    message(WARNING "bench_avx2_blocked: Non-x86_64 platform, AVX2 not available")
endif()

# ============ AVX2 + OpenMP Backend ============
if(OpenMP_CXX_FOUND AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64|X86_64")
    add_executable(bench_avx2_openmp bench/bench_gemm_avx2_openmp.cpp)
    
    target_compile_definitions(bench_avx2_openmp PRIVATE
            EIGEN_NO_DEBUG
    )
    
    target_compile_options(bench_avx2_openmp PRIVATE
            -O3 -DNDEBUG
            -mavx2 -mfma
            -ffast-math -fno-math-errno -fno-trapping-math
    )
    
    target_link_libraries(bench_avx2_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============ CUDA Backend ============
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    add_executable(bench_cuda bench/bench_gemm_cuda.cu)
    
    set_target_properties(bench_cuda PROPERTIES
            CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON
    )
    
    target_compile_definitions(bench_cuda PRIVATE
            EIGEN_NO_DEBUG
            EIGEN_DONT_PARALLELIZE
    )
    
    target_compile_options(bench_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
    
    target_include_directories(bench_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # ============ cuBLAS Backend ============
    find_library(CUBLAS_LIBRARY cublas HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64)
    if(CUBLAS_LIBRARY)
        add_executable(bench_cublas bench/bench_gemm_cublas.cu)
        
        set_target_properties(bench_cublas PROPERTIES
                CUDA_STANDARD 14
                CUDA_STANDARD_REQUIRED ON
        )
        
        target_compile_definitions(bench_cublas PRIVATE
                EIGEN_NO_DEBUG
                EIGEN_DONT_PARALLELIZE
        )
        
        target_compile_options(bench_cublas PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        )
        
        target_include_directories(bench_cublas PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_link_libraries(bench_cublas PRIVATE ${CUBLAS_LIBRARY})
    else()
        message(WARNING "cuBLAS not found, bench_cublas will not be built")
    endif()

    # ============ CUDA v2 (Optimized) Backend ============
    add_executable(bench_cuda_v2 bench/bench_gemm_cuda_v2.cu)
    
    set_target_properties(bench_cuda_v2 PROPERTIES
            CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON
    )
    
    target_compile_definitions(bench_cuda_v2 PRIVATE
            EIGEN_NO_DEBUG
            EIGEN_DONT_PARALLELIZE
    )
    
    target_compile_options(bench_cuda_v2 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
    
    target_include_directories(bench_cuda_v2 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
    message(WARNING "CUDA not found, bench_cuda/bench_cublas/bench_cuda_v2 will not be built")
endif()

# ============ AVX2 + OpenMP + Blocked Backend (Triple Optimization) ============
if(OpenMP_CXX_FOUND AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64|X86_64")
    add_executable(bench_avx2_openmp_blocked bench/bench_gemm_avx2_openmp_blocked.cpp)
    
    target_compile_definitions(bench_avx2_openmp_blocked PRIVATE
            EIGEN_NO_DEBUG
    )
    
    target_compile_options(bench_avx2_openmp_blocked PRIVATE
            -O3 -DNDEBUG
            -mavx2 -mfma
            -ffast-math -fno-math-errno -fno-trapping-math
    )
    
    target_link_libraries(bench_avx2_openmp_blocked PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============ Strassen Algorithm Backend ============
add_executable(bench_strassen bench/bench_gemm_strassen.cpp)

target_compile_definitions(bench_strassen PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_DONT_PARALLELIZE
)

target_compile_options(bench_strassen PRIVATE
        -O3 -DNDEBUG
        -ffast-math -fno-math-errno -fno-trapping-math
)
