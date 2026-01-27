#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_BLOCKED_H
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_BLOCKED_H

#include "macro.hpp"

namespace Auaoalg {
namespace Blocked {

// Cache-friendly blocked/tiled GEMM
// L1 cache ~32KB, L2 cache ~256KB
// Block sizes tuned for modern CPUs

constexpr int BLOCK_M = 64;   // Row block size
constexpr int BLOCK_N = 64;   // Column block size
constexpr int BLOCK_K = 64;   // K dimension block size

// Micro-kernel: compute a small block of C
static AA_ALWAYS_INLINE void MicroKernel(
    const float* AA_RESTRICT A, const float* AA_RESTRICT B,
    float* AA_RESTRICT C, int M, int N, int K,
    int m_start, int m_end, int n_start, int n_end, int k_start, int k_end) {
  
  for (int m = m_start; m < m_end; ++m) {
    const float* AA_RESTRICT a_row = A + m * K;
    float* AA_RESTRICT c_row = C + m * N;
    
    for (int n = n_start; n < n_end; n += 8) {
      int n_max = (n + 8 <= n_end) ? n + 8 : n_end;
      
      // Use small accumulator array for register blocking
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      
      // Load existing C values if not first K block
      if (k_start > 0) {
        for (int nn = 0; nn < n_max - n; ++nn) {
          acc[nn] = c_row[n + nn];
        }
      }
      
      // K-loop with unrolling
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
      
      // Store results
      for (int nn = 0; nn < n_max - n; ++nn) {
        c_row[n + nn] = acc[nn];
      }
    }
  }
}

// A[MxK], B[KxN], C[MxN]
// Uses 3-level loop tiling for better cache utilization
static AA_ALWAYS_INLINE void MatrixMultiply(const float* AA_RESTRICT A,
                                            const float* AA_RESTRICT B,
                                            float* AA_RESTRICT C, int M,
                                            int N, int K) {
  // Initialize C to zero
  for (int i = 0; i < M * N; ++i) {
    C[i] = 0.f;
  }
  
  // Loop over K blocks (outermost for better B reuse)
  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
    
    // Loop over M blocks
    for (int m0 = 0; m0 < M; m0 += BLOCK_M) {
      int m_end = (m0 + BLOCK_M < M) ? m0 + BLOCK_M : M;
      
      // Loop over N blocks
      for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        int n_end = (n0 + BLOCK_N < N) ? n0 + BLOCK_N : N;
        
        // Compute this block
        MicroKernel(A, B, C, M, N, K, m0, m_end, n0, n_end, k0, k_end);
      }
    }
  }
}

}  // namespace Blocked
}  // namespace Auaoalg

#endif
