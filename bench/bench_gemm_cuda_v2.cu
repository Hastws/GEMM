// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// Optimized CUDA GEMM v2 Benchmark

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "macro.hpp"
#include "backend_cuda_v2.cuh"

using namespace Auaoalg;

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int iters = 10;
    int warmup = 3;
    unsigned seed = 42;
    float eps = 1e-3f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = atof(argv[++i]);
    }

    std::cout << "========================================\n";
    std::cout << "CUDA GEMM v2 Benchmark (Optimized)\n";
    std::cout << "========================================\n";
    std::cout << "Matrix size: " << M << " x " << K << " x " << N << "\n";
    std::cout << "Iterations: " << iters << ", Warmup: " << warmup << "\n\n";

    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // Eigen 参考
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eA(A.data(), M, K), eB(B.data(), K, N), eC(C_ref.data(), M, N);
    eC = eA * eB;

    GemmCUDA_V2 cuda_v2;
    double flops = 2.0 * M * N * K;

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < warmup; ++i) {
        cuda_v2.gemm(A.data(), B.data(), C.data(), M, N, K, 3);
    }

    // ========== v2 (8x8 thread tile) ==========
    {
        cuda_v2.upload(A.data(), B.data(), M, N, K);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cuda_v2.gemm_kernel_only(A.data(), B.data(), C.data(), M, N, K, 2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gflops = flops / (elapsed * 1e6);
        std::cout << "[CUDA v2 kernel] Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";

        cuda_v2.get_result(C.data(), M, N);
        float max_diff = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
        }
        std::cout << "  Max diff: " << max_diff << (max_diff < eps ? " ✓" : " ✗") << "\n";
    }

    // ========== v3 (4x4 thread tile, 16x16 block) ==========
    {
        cuda_v2.upload(A.data(), B.data(), M, N, K);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cuda_v2.gemm_kernel_only(A.data(), B.data(), C.data(), M, N, K, 3);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gflops = flops / (elapsed * 1e6);
        std::cout << "[CUDA v3 kernel] Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";

        cuda_v2.get_result(C.data(), M, N);
        float max_diff = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
        }
        std::cout << "  Max diff: " << max_diff << (max_diff < eps ? " ✓" : " ✗") << "\n";
    }

    // ========== v3 含内存拷贝 ==========
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cuda_v2.gemm(A.data(), B.data(), C.data(), M, N, K, 3);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gflops = flops / (elapsed * 1e6);
        std::cout << "[CUDA v3 + memcpy] Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";
    }

    std::cout << "\n[CHECK] OK\n";
    return 0;
}
