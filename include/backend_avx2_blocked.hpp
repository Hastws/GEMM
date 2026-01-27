#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_BLOCKED_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_AVX2_BLOCKED_H

#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include "macro.hpp"

#if !defined(__AVX2__)
#error "AVX2 not enabled: compile with -mavx2 -mfma"
#endif

namespace Auaoalg {
namespace AVX2Blocked {

// Optimized blocking parameters for L1/L2 cache
constexpr int MC = 256;   // M block size (fits in L2)
constexpr int NC = 512;   // N block size (fits in L3)
constexpr int KC = 256;   // K block size (fits in L1)
constexpr int MR = 6;     // Register block M
constexpr int NR = 16;    // Register block N (2x AVX registers)

// Pack A panel [MC x KC] into row-major contiguous buffer
static AA_ALWAYS_INLINE void PackA(const float* AA_RESTRICT A, int lda,
                                   float* AA_RESTRICT Ap, int mc, int kc) {
  for (int i = 0; i < mc; ++i) {
    const float* src = A + i * lda;
    float* dst = Ap + i * kc;
    std::memcpy(dst, src, kc * sizeof(float));
  }
}

// Pack B panel [KC x NC] into column-panel format for efficient access
static AA_ALWAYS_INLINE void PackB(const float* AA_RESTRICT B, int ldb,
                                   float* AA_RESTRICT Bp, int kc, int nc) {
  // Pack as Kc x Nc with NR-wide panels
  for (int j = 0; j < nc; j += NR) {
    int jb = (j + NR <= nc) ? NR : nc - j;
    for (int k = 0; k < kc; ++k) {
      const float* src = B + k * ldb + j;
      float* dst = Bp + (j / NR) * kc * NR + k * NR;
      for (int jj = 0; jj < jb; ++jj) {
        dst[jj] = src[jj];
      }
      for (int jj = jb; jj < NR; ++jj) {
        dst[jj] = 0.f;
      }
    }
  }
}

// 6x16 micro-kernel using AVX2/FMA
static AA_ALWAYS_INLINE void MicroKernel6x16(
    const float* AA_RESTRICT Ap, const float* AA_RESTRICT Bp,
    float* AA_RESTRICT C, int ldc, int kc, int mr, int nr) {
  
  // Accumulators for 6 rows x 16 cols (2 AVX registers per row)
  __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
  __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
  __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
  __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
  __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
  __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
  
  for (int k = 0; k < kc; ++k) {
    __m256 b0 = _mm256_load_ps(Bp + k * NR);
    __m256 b1 = _mm256_load_ps(Bp + k * NR + 8);
    
    __m256 a0 = _mm256_broadcast_ss(Ap + 0 * kc + k);
    c00 = _mm256_fmadd_ps(a0, b0, c00);
    c01 = _mm256_fmadd_ps(a0, b1, c01);
    
    if (mr > 1) {
      __m256 a1 = _mm256_broadcast_ss(Ap + 1 * kc + k);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
    }
    if (mr > 2) {
      __m256 a2 = _mm256_broadcast_ss(Ap + 2 * kc + k);
      c20 = _mm256_fmadd_ps(a2, b0, c20);
      c21 = _mm256_fmadd_ps(a2, b1, c21);
    }
    if (mr > 3) {
      __m256 a3 = _mm256_broadcast_ss(Ap + 3 * kc + k);
      c30 = _mm256_fmadd_ps(a3, b0, c30);
      c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    if (mr > 4) {
      __m256 a4 = _mm256_broadcast_ss(Ap + 4 * kc + k);
      c40 = _mm256_fmadd_ps(a4, b0, c40);
      c41 = _mm256_fmadd_ps(a4, b1, c41);
    }
    if (mr > 5) {
      __m256 a5 = _mm256_broadcast_ss(Ap + 5 * kc + k);
      c50 = _mm256_fmadd_ps(a5, b0, c50);
      c51 = _mm256_fmadd_ps(a5, b1, c51);
    }
  }
  
  // Store with accumulation
  if (nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 0 * ldc);
    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(t0, c00));
  }
  if (nr >= 16) {
    __m256 t1 = _mm256_loadu_ps(C + 0 * ldc + 8);
    _mm256_storeu_ps(C + 0 * ldc + 8, _mm256_add_ps(t1, c01));
  }
  
  if (mr > 1 && nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 1 * ldc);
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(t0, c10));
    if (nr >= 16) {
      __m256 t1 = _mm256_loadu_ps(C + 1 * ldc + 8);
      _mm256_storeu_ps(C + 1 * ldc + 8, _mm256_add_ps(t1, c11));
    }
  }
  if (mr > 2 && nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 2 * ldc);
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(t0, c20));
    if (nr >= 16) {
      __m256 t1 = _mm256_loadu_ps(C + 2 * ldc + 8);
      _mm256_storeu_ps(C + 2 * ldc + 8, _mm256_add_ps(t1, c21));
    }
  }
  if (mr > 3 && nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 3 * ldc);
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(t0, c30));
    if (nr >= 16) {
      __m256 t1 = _mm256_loadu_ps(C + 3 * ldc + 8);
      _mm256_storeu_ps(C + 3 * ldc + 8, _mm256_add_ps(t1, c31));
    }
  }
  if (mr > 4 && nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 4 * ldc);
    _mm256_storeu_ps(C + 4 * ldc, _mm256_add_ps(t0, c40));
    if (nr >= 16) {
      __m256 t1 = _mm256_loadu_ps(C + 4 * ldc + 8);
      _mm256_storeu_ps(C + 4 * ldc + 8, _mm256_add_ps(t1, c41));
    }
  }
  if (mr > 5 && nr >= 8) {
    __m256 t0 = _mm256_loadu_ps(C + 5 * ldc);
    _mm256_storeu_ps(C + 5 * ldc, _mm256_add_ps(t0, c50));
    if (nr >= 16) {
      __m256 t1 = _mm256_loadu_ps(C + 5 * ldc + 8);
      _mm256_storeu_ps(C + 5 * ldc + 8, _mm256_add_ps(t1, c51));
    }
  }
}

// A[MxK], B[KxN], C[MxN]
static void MatrixMultiply(const float* AA_RESTRICT A,
                           const float* AA_RESTRICT B,
                           float* AA_RESTRICT C, int M, int N, int K) {
  // Zero out C
  std::memset(C, 0, M * N * sizeof(float));
  
  // Allocate packing buffers
  float* Ap = (float*)_mm_malloc(MC * KC * sizeof(float), 32);
  float* Bp = (float*)_mm_malloc(KC * NC * sizeof(float), 32);
  
  // 5-loop structure: jc -> pc -> ic -> jr -> ir
  for (int jc = 0; jc < N; jc += NC) {
    int nc = (jc + NC <= N) ? NC : N - jc;
    
    for (int pc = 0; pc < K; pc += KC) {
      int kc = (pc + KC <= K) ? KC : K - pc;
      
      // Pack B panel
      PackB(B + pc * N + jc, N, Bp, kc, nc);
      
      for (int ic = 0; ic < M; ic += MC) {
        int mc = (ic + MC <= M) ? MC : M - ic;
        
        // Pack A panel
        PackA(A + ic * K + pc, K, Ap, mc, kc);
        
        // Micro-kernel loops
        for (int jr = 0; jr < nc; jr += NR) {
          int nr = (jr + NR <= nc) ? NR : nc - jr;
          
          for (int ir = 0; ir < mc; ir += MR) {
            int mr = (ir + MR <= mc) ? MR : mc - ir;
            
            MicroKernel6x16(
              Ap + ir * kc,
              Bp + (jr / NR) * kc * NR,
              C + (ic + ir) * N + jc + jr,
              N, kc, mr, nr
            );
          }
        }
      }
    }
  }
  
  _mm_free(Ap);
  _mm_free(Bp);
}

}  // namespace AVX2Blocked
}  // namespace Auaoalg

#endif
