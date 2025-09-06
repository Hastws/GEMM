#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_H

#include <immintrin.h>

#include "macro.hpp"

#if !defined(__AVX2__)
// 需要在编译器里开启 -mavx2 -mfma
#error "AVX2 not enabled: compile with -mavx2 -mfma"
#endif

namespace Auaoalg {
// Row-major: A[M×K], B[K×N], C[M×N]
// Overwrite: C = A * B

// pack B[j..j+15] (K x 16) into contiguous, 32B-aligned buffer (row-major
// panel)
static AA_ALWAYS_INLINE void pack_B_panel16(const float* AA_RESTRICT B, int N,
                                            int K, int j,
                                            float* AA_RESTRICT Bp) {
  for (int k = 0; k < K; ++k) {
    const float* AA_RESTRICT src = B + k * N + j;
    float* AA_RESTRICT dst = Bp + k * 16;
    __m256 r0 = _mm256_loadu_ps(src + 0);
    __m256 r1 = _mm256_loadu_ps(src + 8);
    _mm256_store_ps(dst + 0, r0);
    _mm256_store_ps(dst + 8, r1);
  }
}

// C[m..m+3, j..j+15] = A[m..m+3, :] * Bp(:, 0..15), Bp packed as Kx16
static AA_ALWAYS_INLINE void kernel_4x16_packed(
    const float* AA_RESTRICT A0, const float* AA_RESTRICT A1,
    const float* AA_RESTRICT A2, const float* AA_RESTRICT A3,
    const float* AA_RESTRICT Bp, int K, float* AA_RESTRICT C0,
    float* AA_RESTRICT C1, float* AA_RESTRICT C2, float* AA_RESTRICT C3,
    int j) {
  const int NR = 16;
  const int V = 8;

  __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
  __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
  __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
  __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

  const float* AA_RESTRICT pB = Bp;
  int k = 0;
  for (; k + 3 < K; k += 4) {
    __m256 b00 = _mm256_load_ps(pB + 0 * NR + 0);
    __m256 b01 = _mm256_load_ps(pB + 0 * NR + V);
    __m256 a0 = _mm256_broadcast_ss(A0 + 0);
    __m256 a1 = _mm256_broadcast_ss(A1 + 0);
    __m256 a2 = _mm256_broadcast_ss(A2 + 0);
    __m256 a3 = _mm256_broadcast_ss(A3 + 0);
    c00 = _mm256_fmadd_ps(a0, b00, c00);
    c01 = _mm256_fmadd_ps(a0, b01, c01);
    c10 = _mm256_fmadd_ps(a1, b00, c10);
    c11 = _mm256_fmadd_ps(a1, b01, c11);
    c20 = _mm256_fmadd_ps(a2, b00, c20);
    c21 = _mm256_fmadd_ps(a2, b01, c21);
    c30 = _mm256_fmadd_ps(a3, b00, c30);
    c31 = _mm256_fmadd_ps(a3, b01, c31);

    __m256 b10 = _mm256_load_ps(pB + 1 * NR + 0);
    __m256 b11 = _mm256_load_ps(pB + 1 * NR + V);
    a0 = _mm256_broadcast_ss(A0 + 1);
    a1 = _mm256_broadcast_ss(A1 + 1);
    a2 = _mm256_broadcast_ss(A2 + 1);
    a3 = _mm256_broadcast_ss(A3 + 1);
    c00 = _mm256_fmadd_ps(a0, b10, c00);
    c01 = _mm256_fmadd_ps(a0, b11, c01);
    c10 = _mm256_fmadd_ps(a1, b10, c10);
    c11 = _mm256_fmadd_ps(a1, b11, c11);
    c20 = _mm256_fmadd_ps(a2, b10, c20);
    c21 = _mm256_fmadd_ps(a2, b11, c21);
    c30 = _mm256_fmadd_ps(a3, b10, c30);
    c31 = _mm256_fmadd_ps(a3, b11, c31);

    __m256 b20 = _mm256_load_ps(pB + 2 * NR + 0);
    __m256 b21 = _mm256_load_ps(pB + 2 * NR + V);
    a0 = _mm256_broadcast_ss(A0 + 2);
    a1 = _mm256_broadcast_ss(A1 + 2);
    a2 = _mm256_broadcast_ss(A2 + 2);
    a3 = _mm256_broadcast_ss(A3 + 2);
    c00 = _mm256_fmadd_ps(a0, b20, c00);
    c01 = _mm256_fmadd_ps(a0, b21, c01);
    c10 = _mm256_fmadd_ps(a1, b20, c10);
    c11 = _mm256_fmadd_ps(a1, b21, c11);
    c20 = _mm256_fmadd_ps(a2, b20, c20);
    c21 = _mm256_fmadd_ps(a2, b21, c21);
    c30 = _mm256_fmadd_ps(a3, b20, c30);
    c31 = _mm256_fmadd_ps(a3, b21, c31);

    __m256 b30 = _mm256_load_ps(pB + 3 * NR + 0);
    __m256 b31 = _mm256_load_ps(pB + 3 * NR + V);
    a0 = _mm256_broadcast_ss(A0 + 3);
    a1 = _mm256_broadcast_ss(A1 + 3);
    a2 = _mm256_broadcast_ss(A2 + 3);
    a3 = _mm256_broadcast_ss(A3 + 3);
    c00 = _mm256_fmadd_ps(a0, b30, c00);
    c01 = _mm256_fmadd_ps(a0, b31, c01);
    c10 = _mm256_fmadd_ps(a1, b30, c10);
    c11 = _mm256_fmadd_ps(a1, b31, c11);
    c20 = _mm256_fmadd_ps(a2, b30, c20);
    c21 = _mm256_fmadd_ps(a2, b31, c21);
    c30 = _mm256_fmadd_ps(a3, b30, c30);
    c31 = _mm256_fmadd_ps(a3, b31, c31);

    A0 += 4;
    A1 += 4;
    A2 += 4;
    A3 += 4;
    pB += 4 * NR;
  }
  for (; k < K; ++k) {
    __m256 b0 = _mm256_load_ps(pB + 0);
    __m256 b1 = _mm256_load_ps(pB + 8);
    __m256 a0 = _mm256_broadcast_ss(A0++);
    __m256 a1 = _mm256_broadcast_ss(A1++);
    __m256 a2 = _mm256_broadcast_ss(A2++);
    __m256 a3 = _mm256_broadcast_ss(A3++);
    c00 = _mm256_fmadd_ps(a0, b0, c00);
    c01 = _mm256_fmadd_ps(a0, b1, c01);
    c10 = _mm256_fmadd_ps(a1, b0, c10);
    c11 = _mm256_fmadd_ps(a1, b1, c11);
    c20 = _mm256_fmadd_ps(a2, b0, c20);
    c21 = _mm256_fmadd_ps(a2, b1, c21);
    c30 = _mm256_fmadd_ps(a3, b0, c30);
    c31 = _mm256_fmadd_ps(a3, b1, c31);
    pB += 16;
  }

  _mm256_storeu_ps(C0 + j + 0, c00);
  _mm256_storeu_ps(C0 + j + 8, c01);
  _mm256_storeu_ps(C1 + j + 0, c10);
  _mm256_storeu_ps(C1 + j + 8, c11);
  _mm256_storeu_ps(C2 + j + 0, c20);
  _mm256_storeu_ps(C2 + j + 8, c21);
  _mm256_storeu_ps(C3 + j + 0, c30);
  _mm256_storeu_ps(C3 + j + 8, c31);
}

// 1x16（packed-B）尾块
static AA_ALWAYS_INLINE void kernel_1x16_packed(const float* AA_RESTRICT A0,
                                                const float* AA_RESTRICT Bp,
                                                int K, float* AA_RESTRICT C0,
                                                int j) {
  const int NR = 16, V = 8;
  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  const float* AA_RESTRICT pB = Bp;
  int k = 0;
  for (; k + 3 < K; k += 4) {
    __m256 b00 = _mm256_load_ps(pB + 0 * NR + 0);
    __m256 b01 = _mm256_load_ps(pB + 0 * NR + V);
    __m256 a0 = _mm256_broadcast_ss(A0 + 0);
    c0 = _mm256_fmadd_ps(a0, b00, c0);
    c1 = _mm256_fmadd_ps(a0, b01, c1);

    __m256 b10 = _mm256_load_ps(pB + 1 * NR + 0);
    __m256 b11 = _mm256_load_ps(pB + 1 * NR + V);
    __m256 a1 = _mm256_broadcast_ss(A0 + 1);
    c0 = _mm256_fmadd_ps(a1, b10, c0);
    c1 = _mm256_fmadd_ps(a1, b11, c1);

    __m256 b20 = _mm256_load_ps(pB + 2 * NR + 0);
    __m256 b21 = _mm256_load_ps(pB + 2 * NR + V);
    __m256 a2 = _mm256_broadcast_ss(A0 + 2);
    c0 = _mm256_fmadd_ps(a2, b20, c0);
    c1 = _mm256_fmadd_ps(a2, b21, c1);

    __m256 b30 = _mm256_load_ps(pB + 3 * NR + 0);
    __m256 b31 = _mm256_load_ps(pB + 3 * NR + V);
    __m256 a3 = _mm256_broadcast_ss(A0 + 3);
    c0 = _mm256_fmadd_ps(a3, b30, c0);
    c1 = _mm256_fmadd_ps(a3, b31, c1);

    A0 += 4;
    pB += 4 * NR;
  }
  for (; k < K; ++k) {
    __m256 b0 = _mm256_load_ps(pB + 0);
    __m256 b1 = _mm256_load_ps(pB + 8);
    __m256 a = _mm256_broadcast_ss(A0++);
    c0 = _mm256_fmadd_ps(a, b0, c0);
    c1 = _mm256_fmadd_ps(a, b1, c1);
    pB += NR;
  }
  _mm256_storeu_ps(C0 + j + 0, c0);
  _mm256_storeu_ps(C0 + j + 8, c1);
}

// 1x8（原矩阵 B，非打包）尾块
static AA_ALWAYS_INLINE void kernel_1x8_unpacked(const float* AA_RESTRICT arow,
                                                 const float* AA_RESTRICT B,
                                                 int N, int K,
                                                 float* AA_RESTRICT crow,
                                                 int j) {
  __m256 c = _mm256_setzero_ps();
  const float* AA_RESTRICT pB = B + j;
  int k = 0;
  for (; k + 3 < K; k += 4) {
    __m256 b0 = _mm256_loadu_ps(pB + 0 * N);
    __m256 a0 = _mm256_broadcast_ss(arow + 0);
    c = _mm256_fmadd_ps(a0, b0, c);

    __m256 b1 = _mm256_loadu_ps(pB + 1 * N);
    __m256 a1 = _mm256_broadcast_ss(arow + 1);
    c = _mm256_fmadd_ps(a1, b1, c);

    __m256 b2 = _mm256_loadu_ps(pB + 2 * N);
    __m256 a2 = _mm256_broadcast_ss(arow + 2);
    c = _mm256_fmadd_ps(a2, b2, c);

    __m256 b3 = _mm256_loadu_ps(pB + 3 * N);
    __m256 a3 = _mm256_broadcast_ss(arow + 3);
    c = _mm256_fmadd_ps(a3, b3, c);

    arow += 4;
    pB += 4 * N;
  }
  for (; k < K; ++k) {
    __m256 b = _mm256_loadu_ps(pB);
    __m256 a = _mm256_broadcast_ss(arow++);
    c = _mm256_fmadd_ps(a, b, c);
    pB += N;
  }
  _mm256_storeu_ps(crow + j, c);
}

// A[MxK], B[KxN], C[MxN], overwrite: C = A * B
static AA_ALWAYS_INLINE void MatrixMultiply(const float* AA_RESTRICT A,
                                            const float* AA_RESTRICT B,
                                            float* AA_RESTRICT C, int M, int N,
                                            int K) {
#if !defined(__AVX2__)
#error "Compile with AVX2/FMA: -mavx2 -mfma (or /arch:AVX2)"
#endif
  const int MR = 4, NR = 16, V = 8;

  const std::size_t pack_bytes = std::size_t(K) * NR * sizeof(float);
  float* AA_RESTRICT Bbuf = (float*)_mm_malloc(pack_bytes, 32);

  int j = 0;
  for (; j + NR <= N; j += NR) {
    pack_B_panel16(B, N, K, j, Bbuf);

    int m = 0;
    for (; m + MR <= M; m += MR) {
      const float* AA_RESTRICT A0 = A + (m + 0) * K;
      const float* AA_RESTRICT A1 = A + (m + 1) * K;
      const float* AA_RESTRICT A2 = A + (m + 2) * K;
      const float* AA_RESTRICT A3 = A + (m + 3) * K;

      float* AA_RESTRICT C0 = C + (m + 0) * N;
      float* AA_RESTRICT C1 = C + (m + 1) * N;
      float* AA_RESTRICT C2 = C + (m + 2) * N;
      float* AA_RESTRICT C3 = C + (m + 3) * N;

      kernel_4x16_packed(A0, A1, A2, A3, Bbuf, K, C0, C1, C2, C3, j);
    }
    for (; m < M; ++m) {
      const float* AA_RESTRICT A0 = A + m * K;
      float* AA_RESTRICT C0 = C + m * N;
      kernel_1x16_packed(A0, Bbuf, K, C0, j);
    }
  }

  _mm_free(Bbuf);

  for (; j + V <= N; j += V) {
    for (int m = 0; m < M; ++m) {
      const float* AA_RESTRICT arow = A + m * K;
      float* AA_RESTRICT crow = C + m * N;
      kernel_1x8_unpacked(arow, B, N, K, crow, j);
    }
  }
  for (; j < N; ++j) {
    for (int m = 0; m < M; ++m) {
      const float* AA_RESTRICT arow = A + m * K;
      float* AA_RESTRICT crow = C + m * N;
      float sum = 0.f;
      for (int k = 0; k < K; ++k) sum += arow[k] * B[k * N + j];
      crow[j] = sum;
    }
  }
}

}  // namespace Auaoalg
#endif
