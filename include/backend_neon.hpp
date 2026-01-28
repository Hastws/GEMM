#pragma once

#if !defined(__ARM_NEON) && !defined(__aarch64__)
// 非 ARM 平台自动退回 REF
#pragma message( \
    "BACKEND_NEON selected but NEON not enabled; falling back to REF")
#include "backend_ref.hpp"
#else
#include <arm_neon.h>

#include <cstring>

#include "macro.hpp"

namespace Auaoalg {

// 4 列向量化（AArch64/ARMv7 NEON；如需 FMA，可替换为 vfmaq_f32）
static AA_ALWAYS_INLINE void MatrixMultiply(const float* AA_RESTRICT A,
                                            const float* AA_RESTRICT B,
                                            float* AA_RESTRICT C, int M, int N,
                                            int K) {
  const int V = 4;
  const int N4 = (N / V) * V;

  for (int i = 0; i < M; ++i) {
    float* crow = C + i * N;
    std::memset(crow, 0, sizeof(float) * N);
    const float* arow = A + i * K;

    // 主体：每次累加 4 列
    for (int j = 0; j < N4; j += V) {
      float32x4_t acc = vdupq_n_f32(0.f);
      for (int k = 0; k < K; ++k) {
        float32x4_t b = vld1q_f32(B + k * N + j);  // B[k, j..j+3]
        float32x4_t as = vdupq_n_f32(arow[k]);     // broadcast A[i,k]
        acc = vmlaq_f32(acc, b, as);               // acc += b * as
      }
      vst1q_f32(crow + j, acc);
    }

    // 尾部：标量
    for (int j = N4; j < N; ++j) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) sum += arow[k] * B[k * N + j];
      crow[j] = sum;
    }
  }
}

}  // namespace Auaoalg

#endif
