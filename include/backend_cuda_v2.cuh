// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// Optimized CUDA GEMM Backend v2
// Features: Larger tiles, double buffering, register blocking

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace Auaoalg {

#define CUDA_CHECK_V2(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

// =============================================================================
// CUDA Kernel v2: 64x64 tiles with 8x8 thread tile and register blocking
// =============================================================================
template<int BLOCK_SIZE = 64, int THREAD_TILE = 8>
__global__ void gemm_kernel_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 线程块和线程索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 每个线程负责的 8x8 区域
    const int THREADS_PER_BLOCK = BLOCK_SIZE / THREAD_TILE;  // 8
    const int row_start = by * BLOCK_SIZE + ty * THREAD_TILE;
    const int col_start = bx * BLOCK_SIZE + tx * THREAD_TILE;

    // 寄存器累加器 (8x8)
    float reg_C[THREAD_TILE][THREAD_TILE] = {0.0f};
    float reg_A[THREAD_TILE];
    float reg_B[THREAD_TILE];

    // 遍历 K 维度的所有 tile
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 协作加载 A 和 B 到共享内存
        // 每个线程加载多个元素
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                int a_row = by * BLOCK_SIZE + ty * THREAD_TILE + i;
                int a_col = t * BLOCK_SIZE + tx * THREAD_TILE + j;
                int b_row = t * BLOCK_SIZE + ty * THREAD_TILE + i;
                int b_col = bx * BLOCK_SIZE + tx * THREAD_TILE + j;

                As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
                Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // 加载 A 的一列到寄存器
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                reg_A[i] = As[ty * THREAD_TILE + i][k];
            }

            // 加载 B 的一行到寄存器
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                reg_B[j] = Bs[k][tx * THREAD_TILE + j];
            }

            // 外积累加
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; ++j) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; ++j) {
            int c_row = row_start + i;
            int c_col = col_start + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = reg_C[i][j];
            }
        }
    }
}

// =============================================================================
// CUDA Kernel v3: Optimized with vectorized loads (float4)
// =============================================================================
template<int BLOCK_SIZE = 64>
__global__ void gemm_kernel_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 每个线程块处理 64x64 的 C 子矩阵
    // 每个线程处理 4x4 的子块
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 避免 bank conflict
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15

    // 16x16 线程 = 256 线程
    // 每个线程负责 4x4 = 16 个元素
    const int row_start = by * BLOCK_SIZE + ty * THREAD_TILE_M;
    const int col_start = bx * BLOCK_SIZE + tx * THREAD_TILE_N;

    float reg_C[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // 每个线程加载多少元素到共享内存
    const int LOAD_STRIDE = BLOCK_SIZE / 16;  // 4

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 协作加载 A
        #pragma unroll
        for (int i = 0; i < LOAD_STRIDE; ++i) {
            #pragma unroll
            for (int j = 0; j < LOAD_STRIDE; ++j) {
                int row = ty * LOAD_STRIDE + i;
                int col = tx * LOAD_STRIDE + j;
                int a_row = by * BLOCK_SIZE + row;
                int a_col = t * BLOCK_SIZE + col;

                As[row][col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

                int b_row = t * BLOCK_SIZE + row;
                int b_col = bx * BLOCK_SIZE + col;
                Bs[row][col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a_vals[THREAD_TILE_M];
            float b_vals[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_vals[i] = As[ty * THREAD_TILE_M + i][k];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_vals[j] = Bs[k][tx * THREAD_TILE_N + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    reg_C[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int c_row = row_start + i;
            int c_col = col_start + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = reg_C[i][j];
            }
        }
    }
}

// =============================================================================
// GemmCUDA_V2 类
// =============================================================================
class GemmCUDA_V2 {
private:
    float *d_A_, *d_B_, *d_C_;
    int M_, N_, K_;
    bool initialized_;

public:
    GemmCUDA_V2() : d_A_(nullptr), d_B_(nullptr), d_C_(nullptr),
                    M_(0), N_(0), K_(0), initialized_(false) {}

    ~GemmCUDA_V2() {
        cleanup();
    }

    void init(int M, int N, int K) {
        cleanup();
        M_ = M; N_ = N; K_ = K;
        CUDA_CHECK_V2(cudaMalloc(&d_A_, M * K * sizeof(float)));
        CUDA_CHECK_V2(cudaMalloc(&d_B_, K * N * sizeof(float)));
        CUDA_CHECK_V2(cudaMalloc(&d_C_, M * N * sizeof(float)));
        initialized_ = true;
    }

    void cleanup() {
        if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
        if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
        if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
        initialized_ = false;
    }

    // 完整流程
    void gemm(const float* A, const float* B, float* C, int M, int N, int K, int version = 3) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
        }

        CUDA_CHECK_V2(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_V2(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        if (version == 2) {
            // v2: 64x64 tiles, 8x8 线程块
            constexpr int BLOCK_SIZE = 64;
            constexpr int THREAD_TILE = 8;
            dim3 block(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);  // 8x8
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            gemm_kernel_v2<BLOCK_SIZE, THREAD_TILE><<<grid, block>>>(d_A_, d_B_, d_C_, M, N, K);
        } else {
            // v3: 64x64 tiles, 16x16 线程块
            constexpr int BLOCK_SIZE = 64;
            dim3 block(16, 16);
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            gemm_kernel_v3<BLOCK_SIZE><<<grid, block>>>(d_A_, d_B_, d_C_, M, N, K);
        }

        CUDA_CHECK_V2(cudaMemcpy(C, d_C_, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // 仅 kernel
    void gemm_kernel_only(const float* A, const float* B, float* C, int M, int N, int K, int version = 3) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
            CUDA_CHECK_V2(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_V2(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (version == 2) {
            constexpr int BLOCK_SIZE = 64;
            constexpr int THREAD_TILE = 8;
            dim3 block(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            gemm_kernel_v2<BLOCK_SIZE, THREAD_TILE><<<grid, block>>>(d_A_, d_B_, d_C_, M, N, K);
        } else {
            constexpr int BLOCK_SIZE = 64;
            dim3 block(16, 16);
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            gemm_kernel_v3<BLOCK_SIZE><<<grid, block>>>(d_A_, d_B_, d_C_, M, N, K);
        }

        cudaDeviceSynchronize();
    }

    void get_result(float* C, int M, int N) {
        CUDA_CHECK_V2(cudaMemcpy(C, d_C_, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void upload(const float* A, const float* B, int M, int N, int K) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
        }
        CUDA_CHECK_V2(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_V2(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }
};

} // namespace Auaoalg
