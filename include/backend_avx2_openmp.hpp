#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_OPENMP_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_OPENMP_H

#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include "macro.hpp"

#if !defined(__AVX2__)
#error "AVX2 not enabled: compile with -mavx2 -mfma"
#endif

namespace Auaoalg {
namespace AVX2OpenMP {

// Cache blocking parameters
constexpr int MC = 128;   // M block
constexpr int NC = 256;   // N block  
constexpr int KC = 256;   // K block
constexpr int MR = 4;     // Register tile M
constexpr int NR = 16;    // Register tile N

// Pack B panel into contiguous memory
static void PackB(const float* AA_RESTRICT B, int ldb,
                  float* AA_RESTRICT Bp, int kc, int nc) {
  for (int j = 0; j < nc; j += NR) {
    int jb = (j + NR <= nc) ? NR : nc - j;
    for (int k = 0; k < kc; ++k) {
      const float* src = B + k * ldb + j;
      float* dst = Bp + (j / NR) * kc * NR + k * NR;
      for (int jj = 0; jj < jb; ++jj) dst[jj] = src[jj];
      for (int jj = jb; jj < NR; ++jj) dst[jj] = 0.f;
    }
  }
}

// 4x16 AVX2 micro-kernel
static AA_ALWAYS_INLINE void MicroKernel4x16(
    const float* AA_RESTRICT A, int lda,
    const float* AA_RESTRICT Bp,
    float* AA_RESTRICT C, int ldc, int kc) {
  
  __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
  __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
  __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
  __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
  
  const float* A0 = A;
  const float* A1 = A + lda;
  const float* A2 = A + 2 * lda;
  const float* A3 = A + 3 * lda;
  
  for (int k = 0; k < kc; ++k) {
    __m256 b0 = _mm256_load_ps(Bp + k * NR);
    __m256 b1 = _mm256_load_ps(Bp + k * NR + 8);
    
    __m256 a0 = _mm256_broadcast_ss(A0 + k);
    __m256 a1 = _mm256_broadcast_ss(A1 + k);
    __m256 a2 = _mm256_broadcast_ss(A2 + k);
    __m256 a3 = _mm256_broadcast_ss(A3 + k);
    
    c00 = _mm256_fmadd_ps(a0, b0, c00);
    c01 = _mm256_fmadd_ps(a0, b1, c01);
    c10 = _mm256_fmadd_ps(a1, b0, c10);
    c11 = _mm256_fmadd_ps(a1, b1, c11);
    c20 = _mm256_fmadd_ps(a2, b0, c20);
    c21 = _mm256_fmadd_ps(a2, b1, c21);
    c30 = _mm256_fmadd_ps(a3, b0, c30);
    c31 = _mm256_fmadd_ps(a3, b1, c31);
  }
  
  // Accumulate to C
  _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc), c00));
  _mm256_storeu_ps(C + 0 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 8), c01));
  _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc), c10));
  _mm256_storeu_ps(C + 1 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 8), c11));
  _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc), c20));
  _mm256_storeu_ps(C + 2 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 8), c21));
  _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc), c30));
  _mm256_storeu_ps(C + 3 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 8), c31));
}

// 1x16 edge case kernel
static AA_ALWAYS_INLINE void MicroKernel1x16(
    const float* AA_RESTRICT A, const float* AA_RESTRICT Bp,
    float* AA_RESTRICT C, int kc) {
  
  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  
  for (int k = 0; k < kc; ++k) {
    __m256 b0 = _mm256_load_ps(Bp + k * NR);
    __m256 b1 = _mm256_load_ps(Bp + k * NR + 8);
    __m256 a = _mm256_broadcast_ss(A + k);
    c0 = _mm256_fmadd_ps(a, b0, c0);
    c1 = _mm256_fmadd_ps(a, b1, c1);
  }
  
  _mm256_storeu_ps(C, _mm256_add_ps(_mm256_loadu_ps(C), c0));
  _mm256_storeu_ps(C + 8, _mm256_add_ps(_mm256_loadu_ps(C + 8), c1));
}

// A[MxK], B[KxN], C[MxN]
static void MatrixMultiply(const float* AA_RESTRICT A,
                           const float* AA_RESTRICT B,
                           float* AA_RESTRICT C, int M, int N, int K,
                           int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = omp_get_max_threads();
  }
  
  // Zero C in parallel
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    C[i] = 0.f;
  }
  
  // Each thread gets its own B packing buffer
  #pragma omp parallel num_threads(num_threads)
  {
    float* Bp = (float*)_mm_malloc(KC * NC * sizeof(float), 32);
    
    // Partition work by N blocks
    #pragma omp for schedule(dynamic)
    for (int jc = 0; jc < N; jc += NC) {
      int nc = (jc + NC <= N) ? NC : N - jc;
      
      for (int pc = 0; pc < K; pc += KC) {
        int kc = (pc + KC <= K) ? KC : K - pc;
        
        PackB(B + pc * N + jc, N, Bp, kc, nc);
        
        for (int ic = 0; ic < M; ic += MC) {
          int mc = (ic + MC <= M) ? MC : M - ic;
          
          for (int jr = 0; jr < nc; jr += NR) {
            for (int ir = 0; ir < mc; ir += MR) {
              int mr = (ir + MR <= mc) ? MR : mc - ir;
              
              if (mr == MR) {
                MicroKernel4x16(
                  A + (ic + ir) * K + pc, K,
                  Bp + (jr / NR) * kc * NR,
                  C + (ic + ir) * N + jc + jr, N, kc
                );
              } else {
                for (int i = 0; i < mr; ++i) {
                  MicroKernel1x16(
                    A + (ic + ir + i) * K + pc,
                    Bp + (jr / NR) * kc * NR,
                    C + (ic + ir + i) * N + jc + jr, kc
                  );
                }
              }
            }
          }
        }
      }
    }
    
    _mm_free(Bp);
  }
}

}  // namespace AVX2OpenMP
}  // namespace Auaoalg

#endif
