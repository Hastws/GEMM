#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_REF_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_REF_H

#include "macro.hpp"

namespace Auaoalg {

static AA_ALWAYS_INLINE void BlockKernel64(const float* AA_RESTRICT a_row,
                                           const float* AA_RESTRICT B,
                                           float* AA_RESTRICT c_row,
                                           const int N, const int K,
                                           const int n) {
  AA_ALIGN(64) float acc[64];
  for (float& t : acc) {
    t = 0.0f;
  }

  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0];
    const float a1 = a_row[k + 1];
    const float a2 = a_row[k + 2];
    const float a3 = a_row[k + 3];

    const float* AA_RESTRICT b0 = B + (k + 0) * N + n;
    const float* AA_RESTRICT b1 = B + (k + 1) * N + n;
    const float* AA_RESTRICT b2 = B + (k + 2) * N + n;
    const float* AA_RESTRICT b3 = B + (k + 3) * N + n;

    for (int tt = 0; tt < 64; tt += 8) {
      float c0 = acc[tt + 0];
      float c1 = acc[tt + 1];
      float c2 = acc[tt + 2];
      float c3 = acc[tt + 3];
      float c4 = acc[tt + 4];
      float c5 = acc[tt + 5];
      float c6 = acc[tt + 6];
      float c7 = acc[tt + 7];

      const float* AA_RESTRICT p0 = b0 + tt;
      const float* AA_RESTRICT p1 = b1 + tt;
      const float* AA_RESTRICT p2 = b2 + tt;
      const float* AA_RESTRICT p3 = b3 + tt;

      c0 += a0 * p0[0] + a1 * p1[0] + a2 * p2[0] + a3 * p3[0];
      c1 += a0 * p0[1] + a1 * p1[1] + a2 * p2[1] + a3 * p3[1];
      c2 += a0 * p0[2] + a1 * p1[2] + a2 * p2[2] + a3 * p3[2];
      c3 += a0 * p0[3] + a1 * p1[3] + a2 * p2[3] + a3 * p3[3];
      c4 += a0 * p0[4] + a1 * p1[4] + a2 * p2[4] + a3 * p3[4];
      c5 += a0 * p0[5] + a1 * p1[5] + a2 * p2[5] + a3 * p3[5];
      c6 += a0 * p0[6] + a1 * p1[6] + a2 * p2[6] + a3 * p3[6];
      c7 += a0 * p0[7] + a1 * p1[7] + a2 * p2[7] + a3 * p3[7];

      acc[tt + 0] = c0;
      acc[tt + 1] = c1;
      acc[tt + 2] = c2;
      acc[tt + 3] = c3;
      acc[tt + 4] = c4;
      acc[tt + 5] = c5;
      acc[tt + 6] = c6;
      acc[tt + 7] = c7;
    }
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* AA_RESTRICT b = B + k * N + n;
    for (int tt = 0; tt < 64; tt += 8) {
      float c0 = acc[tt + 0];
      float c1 = acc[tt + 1];
      float c2 = acc[tt + 2];
      float c3 = acc[tt + 3];
      float c4 = acc[tt + 4];
      float c5 = acc[tt + 5];
      float c6 = acc[tt + 6];
      float c7 = acc[tt + 7];
      const float* AA_RESTRICT p = b + tt;
      c0 += a * p[0];
      c1 += a * p[1];
      c2 += a * p[2];
      c3 += a * p[3];
      c4 += a * p[4];
      c5 += a * p[5];
      c6 += a * p[6];
      c7 += a * p[7];
      acc[tt + 0] = c0;
      acc[tt + 1] = c1;
      acc[tt + 2] = c2;
      acc[tt + 3] = c3;
      acc[tt + 4] = c4;
      acc[tt + 5] = c5;
      acc[tt + 6] = c6;
      acc[tt + 7] = c7;
    }
  }
  for (int t = 0; t < 64; ++t) {
    c_row[n + t] = acc[t];
  }
}

static AA_ALWAYS_INLINE void BlockKernel16(const float* AA_RESTRICT a_row,
                                           const float* AA_RESTRICT B,
                                           float* AA_RESTRICT c_row,
                                           const int N, const int K,
                                           const int n) {
  AA_ALIGN(64) float acc[16];
  for (float& t : acc) t = 0.f;
  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0];
    const float a1 = a_row[k + 1];
    const float a2 = a_row[k + 2];
    const float a3 = a_row[k + 3];
    const float* AA_RESTRICT b0 = B + (k + 0) * N + n;
    const float* AA_RESTRICT b1 = B + (k + 1) * N + n;
    const float* AA_RESTRICT b2 = B + (k + 2) * N + n;
    const float* AA_RESTRICT b3 = B + (k + 3) * N + n;
    for (int t = 0; t < 16; ++t) {
      acc[t] += a0 * b0[t] + a1 * b1[t] + a2 * b2[t] + a3 * b3[t];
    }
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* AA_RESTRICT b = B + k * N + n;
    for (int t = 0; t < 16; ++t) {
      acc[t] += a * b[t];
    }
  }
  for (int t = 0; t < 16; ++t) {
    c_row[n + t] = acc[t];
  }
}

static AA_ALWAYS_INLINE void BlockKernel8(const float* AA_RESTRICT a_row,
                                          const float* AA_RESTRICT B,
                                          float* AA_RESTRICT c_row, const int N,
                                          const int K, const int n) {
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0];
    const float a1 = a_row[k + 1];
    const float a2 = a_row[k + 2];
    const float a3 = a_row[k + 3];
    const float* AA_RESTRICT b0 = B + (k + 0) * N + n;
    const float* AA_RESTRICT b1 = B + (k + 1) * N + n;
    const float* AA_RESTRICT b2 = B + (k + 2) * N + n;
    const float* AA_RESTRICT b3 = B + (k + 3) * N + n;
    c0 += a0 * b0[0] + a1 * b1[0] + a2 * b2[0] + a3 * b3[0];
    c1 += a0 * b0[1] + a1 * b1[1] + a2 * b2[1] + a3 * b3[1];
    c2 += a0 * b0[2] + a1 * b1[2] + a2 * b2[2] + a3 * b3[2];
    c3 += a0 * b0[3] + a1 * b1[3] + a2 * b2[3] + a3 * b3[3];
    c4 += a0 * b0[4] + a1 * b1[4] + a2 * b2[4] + a3 * b3[4];
    c5 += a0 * b0[5] + a1 * b1[5] + a2 * b2[5] + a3 * b3[5];
    c6 += a0 * b0[6] + a1 * b1[6] + a2 * b2[6] + a3 * b3[6];
    c7 += a0 * b0[7] + a1 * b1[7] + a2 * b2[7] + a3 * b3[7];
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* AA_RESTRICT b = B + k * N + n;
    c0 += a * b[0];
    c1 += a * b[1];
    c2 += a * b[2];
    c3 += a * b[3];
    c4 += a * b[4];
    c5 += a * b[5];
    c6 += a * b[6];
    c7 += a * b[7];
  }
  c_row[n + 0] = c0;
  c_row[n + 1] = c1;
  c_row[n + 2] = c2;
  c_row[n + 3] = c3;
  c_row[n + 4] = c4;
  c_row[n + 5] = c5;
  c_row[n + 6] = c6;
  c_row[n + 7] = c7;
}

// A[MxK], B[KxN], C[MxN]
static AA_ALWAYS_INLINE void MatrixMultiply(const float* AA_RESTRICT A,
                                            const float* AA_RESTRICT B,
                                            float* AA_RESTRICT C, const int M,
                                            const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    const float* AA_RESTRICT a_row = A + m * K;
    float* AA_RESTRICT c_row = C + m * N;

    int n = 0;
    for (; n + 64 <= N; n += 64) {
      BlockKernel64(a_row, B, c_row, N, K, n);
    }
    for (; n + 16 <= N; n += 16) {
      BlockKernel16(a_row, B, c_row, N, K, n);
    }
    for (; n + 8 <= N; n += 8) {
      BlockKernel8(a_row, B, c_row, N, K, n);
    }

    for (; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) sum += a_row[k] * B[k * N + n];
      c_row[n] = sum;
    }
  }
}

// A[MxK], B[KxN], c[MxN]
static AA_ALWAYS_INLINE void MatrixMultiplyMKN(const float* A, const float* B,
                                               float* C, const int M,
                                               const int N, const int K) {
  float amk = 0.0;
  for (int m = 0; m < M; m++) {
    const float* a_row = A + m * K;
    float* c_row = C + m * N;
    for (int k = 0; k < K; k++) {
      const float* b_row = B + k * N;
      amk = a_row[k];
      for (int n = 0; n < N; n++) {
        c_row[n] += amk * b_row[n];
      }
    }
  }
}

// A[Mxk], B[KxN], C[MxN]
static AA_ALWAYS_INLINE void MatrixMultiplyNaive2(const float* A,
                                                  const float* B, float* C,
                                                  const int M, const int N,
                                                  const int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

// A[Mxk], B[KxN], C[MxN]
static AA_ALWAYS_INLINE void MatrixMultiplyNaive(const float* A, const float* B,
                                                 float* C, const int M,
                                                 const int N, const int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

}  // namespace Auaoalg
#endif
