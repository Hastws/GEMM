// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// cuBLAS GEMM Backend - NVIDIA's optimized BLAS library
// This represents the theoretical peak performance on NVIDIA GPUs

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>

namespace Auaoalg {

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

#define CUDA_CHECK_CUBLAS(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

class GemmCuBLAS {
private:
    cublasHandle_t handle_;
    float *d_A_, *d_B_, *d_C_;
    int M_, N_, K_;
    bool initialized_;

public:
    GemmCuBLAS() : handle_(nullptr), d_A_(nullptr), d_B_(nullptr), d_C_(nullptr),
                   M_(0), N_(0), K_(0), initialized_(false) {}

    ~GemmCuBLAS() {
        cleanup();
    }

    void init(int M, int N, int K) {
        cleanup();
        M_ = M; N_ = N; K_ = K;

        CUBLAS_CHECK(cublasCreate(&handle_));

        // 分配设备内存
        CUDA_CHECK_CUBLAS(cudaMalloc(&d_A_, M * K * sizeof(float)));
        CUDA_CHECK_CUBLAS(cudaMalloc(&d_B_, K * N * sizeof(float)));
        CUDA_CHECK_CUBLAS(cudaMalloc(&d_C_, M * N * sizeof(float)));

        initialized_ = true;
    }

    void cleanup() {
        if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
        if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
        if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
        if (handle_) { cublasDestroy(handle_); handle_ = nullptr; }
        initialized_ = false;
    }

    // 完整流程（含内存拷贝）
    void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
        }

        // H2D
        CUDA_CHECK_CUBLAS(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_CUBLAS(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        // cuBLAS SGEMM: C = alpha * A * B + beta * C
        // 注意：cuBLAS 使用列主序，我们使用行主序
        // 所以计算 C^T = B^T * A^T，即 cuBLAS(B, A) with leading dimensions
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Row-major: C[M,N] = A[M,K] * B[K,N]
        // cuBLAS (col-major): C^T[N,M] = B^T[N,K] * A^T[K,M]
        CUBLAS_CHECK(cublasSgemm(handle_,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B_, N,  // B as row-major = B^T in col-major
                                  d_A_, K,  // A as row-major = A^T in col-major
                                  &beta,
                                  d_C_, N));

        // D2H
        CUDA_CHECK_CUBLAS(cudaMemcpy(C, d_C_, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // 仅计算（用于 benchmark kernel 性能）
    void gemm_kernel_only(const float* A, const float* B, float* C, int M, int N, int K) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
            // 首次需要拷贝数据
            CUDA_CHECK_CUBLAS(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_CUBLAS(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemm(handle_,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B_, N,
                                  d_A_, K,
                                  &beta,
                                  d_C_, N));

        cudaDeviceSynchronize();
    }

    // 获取结果（用于验证）
    void get_result(float* C, int M, int N) {
        CUDA_CHECK_CUBLAS(cudaMemcpy(C, d_C_, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // 上传数据（用于 benchmark 时分离内存传输）
    void upload(const float* A, const float* B, int M, int N, int K) {
        if (!initialized_ || M != M_ || N != N_ || K != K_) {
            init(M, N, K);
        }
        CUDA_CHECK_CUBLAS(cudaMemcpy(d_A_, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_CUBLAS(cudaMemcpy(d_B_, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }
};

} // namespace Auaoalg
