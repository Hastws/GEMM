#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_OPENMP_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_OPENMP_H

#include <omp.h>
#include "macro.hpp"

namespace Auaoalg {
namespace OpenMP {

// OpenMP parallelized GEMM with blocking
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 256;

// Micro-kernel for a single block
static AA_ALWAYS_INLINE void MicroKernel(
    const float* AA_RESTRICT A, const float* AA_RESTRICT B,
    float* AA_RESTRICT C, int N, int K,
    int m_start, int m_end, int n_start, int n_end, int k_start, int k_end) {
  
  for (int m = m_start; m < m_end; ++m) {
    const float* AA_RESTRICT a_row = A + m * K;
    float* AA_RESTRICT c_row = C + m * N;
    
    for (int n = n_start; n < n_end; n += 8) {
      int n_max = (n + 8 <= n_end) ? n + 8 : n_end;
      
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      
      if (k_start > 0) {
        for (int nn = 0; nn < n_max - n; ++nn) {
          acc[nn] = c_row[n + nn];
        }
      }
      
      int k = k_start;
      for (; k + 4 <= k_end; k += 4) {
        const float a0 = a_row[k + 0];
        const float a1 = a_row[k + 1];
        const float a2 = a_row[k + 2];
        const float a3 = a_row[k + 3];
        
        const float* AA_RESTRICT b0 = B + (k + 0) * N + n;
        const float* AA_RESTRICT b1 = B + (k + 1) * N + n;
        const float* AA_RESTRICT b2 = B + (k + 2) * N + n;
        const float* AA_RESTRICT b3 = B + (k + 3) * N + n;
        
        for (int nn = 0; nn < n_max - n; ++nn) {
          acc[nn] += a0 * b0[nn] + a1 * b1[nn] + a2 * b2[nn] + a3 * b3[nn];
        }
      }
      
      for (; k < k_end; ++k) {
        const float a = a_row[k];
        const float* AA_RESTRICT b = B + k * N + n;
        for (int nn = 0; nn < n_max - n; ++nn) {
          acc[nn] += a * b[nn];
        }
      }
      
      for (int nn = 0; nn < n_max - n; ++nn) {
        c_row[n + nn] = acc[nn];
      }
    }
  }
}

// A[MxK], B[KxN], C[MxN]
static AA_ALWAYS_INLINE void MatrixMultiply(const float* AA_RESTRICT A,
                                            const float* AA_RESTRICT B,
                                            float* AA_RESTRICT C, int M,
                                            int N, int K, int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = omp_get_max_threads();
  }
  
  // Initialize C to zero (parallel)
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    C[i] = 0.f;
  }
  
  // Compute number of M blocks
  int num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
  
  // Parallelize over M blocks and K blocks
  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int mb = 0; mb < num_m_blocks; ++mb) {
      int m0 = mb * BLOCK_M;
      int m_end = (m0 + BLOCK_M < M) ? m0 + BLOCK_M : M;
      
      for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        int n_end = (n0 + BLOCK_N < N) ? n0 + BLOCK_N : N;
        MicroKernel(A, B, C, N, K, m0, m_end, n0, n_end, k0, k_end);
      }
    }
  }
}

}  // namespace OpenMP
}  // namespace Auaoalg

#endif
