// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// AVX2 + OpenMP + Blocked GEMM Backend
// Triple optimization: SIMD + Multi-threading + Cache Tiling

#pragma once

#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace Auaoalg {

// 分块参数 - 针对 L1/L2/L3 缓存优化
constexpr int MC_TRIPLE = 128;   // M 方向分块 (fits in L2)
constexpr int NC_TRIPLE = 256;   // N 方向分块 (fits in L3)
constexpr int KC_TRIPLE = 256;   // K 方向分块 (fits in L1)
constexpr int MR_TRIPLE = 8;     // micro-kernel M
constexpr int NR_TRIPLE = 8;     // micro-kernel N

// 8x8 micro-kernel using AVX2
inline void micro_kernel_8x8_avx2(
    int K,
    const float* __restrict__ A,  // packed A: MR x K
    const float* __restrict__ B,  // packed B: K x NR
    float* __restrict__ C,
    int ldc
) {
    // 8x8 累加器 (每行用一个 __m256)
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    for (int p = 0; p < K; ++p) {
        // 加载 B 的 8 个元素
        __m256 b_vec = _mm256_loadu_ps(B + p * NR_TRIPLE);

        // 广播 A 的每个元素并累加
        __m256 a0 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 0);
        __m256 a1 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 1);
        __m256 a2 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 2);
        __m256 a3 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 3);
        __m256 a4 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 4);
        __m256 a5 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 5);
        __m256 a6 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 6);
        __m256 a7 = _mm256_broadcast_ss(A + p * MR_TRIPLE + 7);

        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
        c4 = _mm256_fmadd_ps(a4, b_vec, c4);
        c5 = _mm256_fmadd_ps(a5, b_vec, c5);
        c6 = _mm256_fmadd_ps(a6, b_vec, c6);
        c7 = _mm256_fmadd_ps(a7, b_vec, c7);
    }

    // 写回 C (累加模式)
    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc), c0));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc), c1));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc), c2));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc), c3));
    _mm256_storeu_ps(C + 4 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc), c4));
    _mm256_storeu_ps(C + 5 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc), c5));
    _mm256_storeu_ps(C + 6 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 6 * ldc), c6));
    _mm256_storeu_ps(C + 7 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 7 * ldc), c7));
}

// Pack A: 将 A 的 mc x kc 块打包为 MR x kc 的连续内存
inline void pack_A_triple(
    const float* __restrict__ A, int lda,
    float* __restrict__ packed_A,
    int mc, int kc
) {
    for (int i = 0; i < mc; i += MR_TRIPLE) {
        int mr = std::min(MR_TRIPLE, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed_A[p * MR_TRIPLE + ii] = A[(i + ii) * lda + p];
            }
            // 填充零（如果 mr < MR）
            for (int ii = mr; ii < MR_TRIPLE; ++ii) {
                packed_A[p * MR_TRIPLE + ii] = 0.0f;
            }
        }
        packed_A += kc * MR_TRIPLE;
    }
}

// Pack B: 将 B 的 kc x nc 块打包为 kc x NR 的连续内存
inline void pack_B_triple(
    const float* __restrict__ B, int ldb,
    float* __restrict__ packed_B,
    int kc, int nc
) {
    for (int j = 0; j < nc; j += NR_TRIPLE) {
        int nr = std::min(NR_TRIPLE, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed_B[p * NR_TRIPLE + jj] = B[p * ldb + j + jj];
            }
            for (int jj = nr; jj < NR_TRIPLE; ++jj) {
                packed_B[p * NR_TRIPLE + jj] = 0.0f;
            }
        }
        packed_B += kc * NR_TRIPLE;
    }
}

// 主函数
inline void gemm_avx2_openmp_blocked(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 初始化 C 为零
    std::memset(C, 0, M * N * sizeof(float));

    int num_threads = omp_get_max_threads();

    // 预分配每个线程的打包缓冲区
    float** packed_A_buffers = new float*[num_threads];
    float** packed_B_buffers = new float*[num_threads];

    for (int t = 0; t < num_threads; ++t) {
        packed_A_buffers[t] = (float*)aligned_alloc(64, MC_TRIPLE * KC_TRIPLE * sizeof(float));
        packed_B_buffers[t] = (float*)aligned_alloc(64, KC_TRIPLE * NC_TRIPLE * sizeof(float));
    }

    // 5层循环结构 (BLIS style)
    // Loop 5: K 方向分块
    for (int pc = 0; pc < K; pc += KC_TRIPLE) {
        int kc = std::min(KC_TRIPLE, K - pc);

        // Loop 4: N 方向分块 (并行化)
        #pragma omp parallel for schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC_TRIPLE) {
            int nc = std::min(NC_TRIPLE, N - jc);
            int tid = omp_get_thread_num();

            // Pack B panel
            pack_B_triple(B + pc * N + jc, N, packed_B_buffers[tid], kc, nc);

            // Loop 3: M 方向分块
            for (int ic = 0; ic < M; ic += MC_TRIPLE) {
                int mc = std::min(MC_TRIPLE, M - ic);

                // Pack A panel
                pack_A_triple(A + ic * K + pc, K, packed_A_buffers[tid], mc, kc);

                // Loop 2 & 1: micro-kernel 计算
                float* packed_A_ptr = packed_A_buffers[tid];
                for (int ir = 0; ir < mc; ir += MR_TRIPLE) {
                    int mr = std::min(MR_TRIPLE, mc - ir);

                    float* packed_B_ptr = packed_B_buffers[tid];
                    for (int jr = 0; jr < nc; jr += NR_TRIPLE) {
                        int nr = std::min(NR_TRIPLE, nc - jr);

                        // 调用 micro-kernel
                        if (mr == MR_TRIPLE && nr == NR_TRIPLE) {
                            micro_kernel_8x8_avx2(
                                kc,
                                packed_A_ptr,
                                packed_B_ptr,
                                C + (ic + ir) * N + (jc + jr),
                                N
                            );
                        } else {
                            // 边界情况：使用标量处理
                            for (int i = 0; i < mr; ++i) {
                                for (int j = 0; j < nr; ++j) {
                                    float sum = 0.0f;
                                    for (int p = 0; p < kc; ++p) {
                                        sum += packed_A_ptr[p * MR_TRIPLE + i] * 
                                               packed_B_ptr[p * NR_TRIPLE + j];
                                    }
                                    C[(ic + ir + i) * N + (jc + jr + j)] += sum;
                                }
                            }
                        }

                        packed_B_ptr += kc * NR_TRIPLE;
                    }
                    packed_A_ptr += kc * MR_TRIPLE;
                }
            }
        }
    }

    // 清理
    for (int t = 0; t < num_threads; ++t) {
        free(packed_A_buffers[t]);
        free(packed_B_buffers[t]);
    }
    delete[] packed_A_buffers;
    delete[] packed_B_buffers;
}

} // namespace Auaoalg
