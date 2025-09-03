#pragma once

#if !defined(__AVX2__)
#pragma message( \
    "BACKEND_AVX2 selected but __AVX2__ not enabled; falling back to REF")
#include "backend_ref.hpp"

#else

#include <immintrin.h>

#include <cstring>

namespace autoalg {
// 简单 8 列向量化（带尾处理）
// 处理 16 列（两个累加器），k 展开 4
static inline void MatrixMultiply(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
  const int V = 8;
  const int N16 = (N / (2 * V)) * (2 * V);
  for (int i = 0; i < M; ++i) {
    const float* arow = A + i * K;
    float* crow = C + i * N;

    int j = 0;
    for (; j < N16; j += 2 * V) {
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();

      int k = 0;
      for (; k + 3 < K; k += 4) {
        __m256 a0 = _mm256_set1_ps(arow[k + 0]);
        __m256 b00 = _mm256_loadu_ps(B + (k + 0) * N + j);
        __m256 b01 = _mm256_loadu_ps(B + (k + 0) * N + j + V);
        acc0 = _mm256_fmadd_ps(a0, b00, acc0);
        acc1 = _mm256_fmadd_ps(a0, b01, acc1);

        __m256 a1 = _mm256_set1_ps(arow[k + 1]);
        __m256 b10 = _mm256_loadu_ps(B + (k + 1) * N + j);
        __m256 b11 = _mm256_loadu_ps(B + (k + 1) * N + j + V);
        acc0 = _mm256_fmadd_ps(a1, b10, acc0);
        acc1 = _mm256_fmadd_ps(a1, b11, acc1);

        __m256 a2 = _mm256_set1_ps(arow[k + 2]);
        __m256 b20 = _mm256_loadu_ps(B + (k + 2) * N + j);
        __m256 b21 = _mm256_loadu_ps(B + (k + 2) * N + j + V);
        acc0 = _mm256_fmadd_ps(a2, b20, acc0);
        acc1 = _mm256_fmadd_ps(a2, b21, acc1);

        __m256 a3 = _mm256_set1_ps(arow[k + 3]);
        __m256 b30 = _mm256_loadu_ps(B + (k + 3) * N + j);
        __m256 b31 = _mm256_loadu_ps(B + (k + 3) * N + j + V);
        acc0 = _mm256_fmadd_ps(a3, b30, acc0);
        acc1 = _mm256_fmadd_ps(a3, b31, acc1);
      }
      for (; k < K; ++k) {
        __m256 a = _mm256_set1_ps(arow[k]);
        __m256 b0 = _mm256_loadu_ps(B + k * N + j);
        __m256 b1 = _mm256_loadu_ps(B + k * N + j + V);
        acc0 = _mm256_fmadd_ps(a, b0, acc0);
        acc1 = _mm256_fmadd_ps(a, b1, acc1);
      }

      _mm256_storeu_ps(crow + j, acc0);
      _mm256_storeu_ps(crow + j + V, acc1);
    }

    // 还可以保留你原来的 8 列块，然后再做标量尾部
    for (; j + V <= N; j += V) {
      __m256 acc = _mm256_setzero_ps();
      for (int k = 0; k < K; ++k) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(arow[k]),
                              _mm256_loadu_ps(B + k * N + j), acc);
      }
      _mm256_storeu_ps(crow + j, acc);
    }
    for (; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += arow[k] * B[k * N + j];
      crow[j] = s;
    }
  }
}

}  // namespace autoalg
#endif
