// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// Strassen Algorithm GEMM Backend
// Divide-and-conquer matrix multiplication with O(n^2.807) complexity
// Faster than O(n^3) for large matrices

#pragma once

#include <cstring>
#include <algorithm>
#include <cmath>

namespace Auaoalg {

// Strassen 算法的截止阈值
// 当矩阵小于此阈值时，使用标准算法
constexpr int STRASSEN_THRESHOLD = 64;

// 辅助函数：矩阵加法 C = A + B
inline void matrix_add(
    const float* A, const float* B, float* C,
    int M, int N, int lda, int ldb, int ldc
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
    }
}

// 辅助函数：矩阵减法 C = A - B
inline void matrix_sub(
    const float* A, const float* B, float* C,
    int M, int N, int lda, int ldb, int ldc
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
        }
    }
}

// 标准矩阵乘法（用于小矩阵）
inline void gemm_standard(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// Strassen 递归实现
// 假设矩阵是方阵且大小为 2 的幂次
void strassen_recursive(
    const float* A, const float* B, float* C,
    int n, int lda, int ldb, int ldc,
    float* workspace
) {
    // 基础情况
    if (n <= STRASSEN_THRESHOLD) {
        gemm_standard(A, B, C, n, n, n, lda, ldb, ldc);
        return;
    }

    int half = n / 2;
    int ws_size = half * half;

    // 分块指针
    const float* A11 = A;
    const float* A12 = A + half;
    const float* A21 = A + half * lda;
    const float* A22 = A + half * lda + half;

    const float* B11 = B;
    const float* B12 = B + half;
    const float* B21 = B + half * ldb;
    const float* B22 = B + half * ldb + half;

    float* C11 = C;
    float* C12 = C + half;
    float* C21 = C + half * ldc;
    float* C22 = C + half * ldc + half;

    // 工作空间分配
    float* M1 = workspace;
    float* M2 = workspace + ws_size;
    float* M3 = workspace + 2 * ws_size;
    float* M4 = workspace + 3 * ws_size;
    float* M5 = workspace + 4 * ws_size;
    float* M6 = workspace + 5 * ws_size;
    float* M7 = workspace + 6 * ws_size;
    float* T1 = workspace + 7 * ws_size;
    float* T2 = workspace + 8 * ws_size;
    float* next_ws = workspace + 9 * ws_size;

    // Strassen 的 7 次乘法
    
    // M1 = (A11 + A22) * (B11 + B22)
    matrix_add(A11, A22, T1, half, half, lda, lda, half);
    matrix_add(B11, B22, T2, half, half, ldb, ldb, half);
    strassen_recursive(T1, T2, M1, half, half, half, half, next_ws);

    // M2 = (A21 + A22) * B11
    matrix_add(A21, A22, T1, half, half, lda, lda, half);
    strassen_recursive(T1, B11, M2, half, half, ldb, half, next_ws);

    // M3 = A11 * (B12 - B22)
    matrix_sub(B12, B22, T1, half, half, ldb, ldb, half);
    strassen_recursive(A11, T1, M3, half, lda, half, half, next_ws);

    // M4 = A22 * (B21 - B11)
    matrix_sub(B21, B11, T1, half, half, ldb, ldb, half);
    strassen_recursive(A22, T1, M4, half, lda, half, half, next_ws);

    // M5 = (A11 + A12) * B22
    matrix_add(A11, A12, T1, half, half, lda, lda, half);
    strassen_recursive(T1, B22, M5, half, half, ldb, half, next_ws);

    // M6 = (A21 - A11) * (B11 + B12)
    matrix_sub(A21, A11, T1, half, half, lda, lda, half);
    matrix_add(B11, B12, T2, half, half, ldb, ldb, half);
    strassen_recursive(T1, T2, M6, half, half, half, half, next_ws);

    // M7 = (A12 - A22) * (B21 + B22)
    matrix_sub(A12, A22, T1, half, half, lda, lda, half);
    matrix_add(B21, B22, T2, half, half, ldb, ldb, half);
    strassen_recursive(T1, T2, M7, half, half, half, half, next_ws);

    // 组合结果
    // C11 = M1 + M4 - M5 + M7
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C11[i * ldc + j] = M1[i * half + j] + M4[i * half + j] 
                             - M5[i * half + j] + M7[i * half + j];
        }
    }

    // C12 = M3 + M5
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C12[i * ldc + j] = M3[i * half + j] + M5[i * half + j];
        }
    }

    // C21 = M2 + M4
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C21[i * ldc + j] = M2[i * half + j] + M4[i * half + j];
        }
    }

    // C22 = M1 - M2 + M3 + M6
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C22[i * ldc + j] = M1[i * half + j] - M2[i * half + j] 
                             + M3[i * half + j] + M6[i * half + j];
        }
    }
}

// 计算需要的工作空间大小
inline size_t strassen_workspace_size(int n) {
    if (n <= STRASSEN_THRESHOLD) return 0;
    
    int half = n / 2;
    // 9 个 half x half 矩阵 + 递归所需空间
    return 9 * half * half + strassen_workspace_size(half);
}

// 将非 2 的幂次矩阵扩展到 2 的幂次
inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// 主函数
inline void gemm_strassen(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 对于非方阵或非 2 的幂次，需要填充
    int max_dim = std::max({M, N, K});
    int padded_size = next_power_of_2(max_dim);

    // 如果太小，直接用标准算法
    if (padded_size <= STRASSEN_THRESHOLD) {
        std::memset(C, 0, M * N * sizeof(float));
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a_ik = A[i * K + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
        return;
    }

    // 分配填充后的矩阵
    float* A_pad = new float[padded_size * padded_size]();
    float* B_pad = new float[padded_size * padded_size]();
    float* C_pad = new float[padded_size * padded_size]();

    // 复制数据
    for (int i = 0; i < M; ++i) {
        std::memcpy(A_pad + i * padded_size, A + i * K, K * sizeof(float));
    }
    for (int i = 0; i < K; ++i) {
        std::memcpy(B_pad + i * padded_size, B + i * N, N * sizeof(float));
    }

    // 分配工作空间
    size_t ws_size = strassen_workspace_size(padded_size);
    float* workspace = new float[ws_size];

    // 执行 Strassen
    strassen_recursive(A_pad, B_pad, C_pad, padded_size, 
                       padded_size, padded_size, padded_size, workspace);

    // 复制结果
    for (int i = 0; i < M; ++i) {
        std::memcpy(C + i * N, C_pad + i * padded_size, N * sizeof(float));
    }

    delete[] A_pad;
    delete[] B_pad;
    delete[] C_pad;
    delete[] workspace;
}

} // namespace Auaoalg
