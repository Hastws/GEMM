#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_REF_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_REF_H

namespace autoalg {

static inline void BlockKernel64(const float* a_row, const float* B,
                                 float* c_row, const int N, const int K,
                                 const int j) {
  float acc[64];
  for (int t = 0; t < 64; ++t) {
    acc[t] = 0.0f;
  }

  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0];
    const float a1 = a_row[k + 1];
    const float a2 = a_row[k + 2];
    const float a3 = a_row[k + 3];

    const float* b0 = B + (k + 0) * N + j;
    const float* b1 = B + (k + 1) * N + j;
    const float* b2 = B + (k + 2) * N + j;
    const float* b3 = B + (k + 3) * N + j;

    for (int t = 0; t < 64; ++t) {
      acc[t] += a0 * b0[t] + a1 * b1[t] + a2 * b2[t] + a3 * b3[t];
    }
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* b = B + k * N + j;
    for (int t = 0; t < 64; ++t) {
      acc[t] += a * b[t];
    }
  }

  for (int t = 0; t < 64; ++t) {
    c_row[j + t] = acc[t];
  }
}

static inline void BlockKernel16(const float* a_row, const float* B,
                                 float* c_row, const int N, const int K,
                                 const int j) {
  float acc[16];
  for (int t = 0; t < 16; ++t) {
    acc[t] = 0.f;
  }

  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0], a1 = a_row[k + 1];
    const float a2 = a_row[k + 2], a3 = a_row[k + 3];
    const float* b0 = B + (k + 0) * N + j;
    const float* b1 = B + (k + 1) * N + j;
    const float* b2 = B + (k + 2) * N + j;
    const float* b3 = B + (k + 3) * N + j;
    for (int t = 0; t < 16; ++t)
      acc[t] += a0 * b0[t] + a1 * b1[t] + a2 * b2[t] + a3 * b3[t];
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* b = B + k * N + j;
    for (int t = 0; t < 16; ++t) {
      acc[t] += a * b[t];
    }
  }

  for (int t = 0; t < 16; ++t) {
    c_row[j + t] = acc[t];
  }
}

static inline void BlockKernel8(const float* a_row, const float* B,
                                float* c_row, const int N, const int K,
                                const int j) {
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  int k = 0;
  for (; k + 3 < K; k += 4) {
    const float a0 = a_row[k + 0], a1 = a_row[k + 1];
    const float a2 = a_row[k + 2], a3 = a_row[k + 3];
    const float* b0 = B + (k + 0) * N + j;
    const float* b1 = B + (k + 1) * N + j;
    const float* b2 = B + (k + 2) * N + j;
    const float* b3 = B + (k + 3) * N + j;
    for (int s = 0; s < 8; ++s)
      acc[s] += a0 * b0[s] + a1 * b1[s] + a2 * b2[s] + a3 * b3[s];
  }
  for (; k < K; ++k) {
    const float a = a_row[k];
    const float* b = B + k * N + j;
    for (int s = 0; s < 8; ++s) {
      acc[s] += a * b[s];
    }
  }

  for (int s = 0; s < 8; ++s) {
    c_row[j + s] = acc[s];
  }
}

// A[MxK], B[KxN], c[MxN]
static inline void MatrixMultiply(const float* A, const float* B, float* C,
                                  const int M, const int N, const int K) {
  for (int i = 0; i < M; ++i) {
    const float* a_row = A + i * K;
    float* c_row = C + i * N;

    int j = 0;
    for (; j + 64 <= N; j += 64) {
      BlockKernel64(a_row, B, c_row, N, K, j);
    }
    for (; j + 16 <= N; j += 16) {
      BlockKernel16(a_row, B, c_row, N, K, j);
    }
    for (; j + 8 <= N; j += 8) {
      BlockKernel8(a_row, B, c_row, N, K, j);
    }

    for (; j < N; ++j) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += a_row[k] * B[k * N + j];
      }
      c_row[j] = sum;
    }
  }
}

// <!----------------------------------------------------------------------->

// // A[MxK], B[KxN], c[MxN]
// static inline void MatrixMultiply(const float* A, const float* B, float* C,
//                                   const int M, const int N, const int K) {
//   float amk = 0.0;
//   for (int m = 0; m < M; m++) {
//     const float* a_row = A + m * K;
//     float* c_row = C + m * N;
//     for (int k = 0; k < K; k++) {
//       const float* b_row = B + k * N;
//       amk = a_row[k];
//       for (int n = 0; n < N; n++) {
//         c_row[n] += amk * b_row[n];
//       }
//     }
//   }
// }

// <!----------------------------------------------------------------------->

// // A[Mxk], B[KxN], C[MxN]
// static inline void MatrixMultiply(const float* A, const float* B, float* C,
//                                   const int M, const int N, const int K) {
//   for (int m = 0; m < M; m++) {
//     for (int n = 0; n < N; n++) {
//       float sum = 0;
//       for (int k = 0; k < K; k++) {
//         sum += A[m * K + k] * B[k * N + n];
//       }
//       C[m * N + n] = sum;
//     }
//   }
// }

// <!----------------------------------------------------------------------->

// // A[Mxk], B[KxN], C[MxN]
// static inline void MatrixMultiply(const float* A, const float* B, float* C,
//                                   const int M, const int N, const int K) {
//   for (int m = 0; m < M; m++) {
//     for (int n = 0; n < N; n++) {
//       for (int k = 0; k < K; k++) {
//         C[m * N + n] += A[m * K + k] * B[k * N + n];
//       }
//     }
//   }
// }

}  // namespace autoalg
#endif
