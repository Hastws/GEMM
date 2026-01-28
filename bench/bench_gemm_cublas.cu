// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// cuBLAS GEMM Benchmark

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "macro.hpp"
#include "backend_cublas.cuh"

using namespace Auaoalg;

int main(int argc, char* argv[]) {
    // 默认参数
    int M = 1024, N = 1024, K = 1024;
    int iters = 10;
    int warmup = 3;
    unsigned seed = 42;
    float eps = 1e-3f;

    // 解析命令行参数
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
    std::cout << "cuBLAS GEMM Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix size: " << M << " x " << K << " x " << N << "\n";
    std::cout << "Iterations: " << iters << ", Warmup: " << warmup << "\n\n";

    // 分配内存
    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);

    // 初始化
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // Eigen 参考计算
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eA(A.data(), M, K), eB(B.data(), K, N), eC(C_ref.data(), M, N);
    eC = eA * eB;

    // cuBLAS
    GemmCuBLAS cublas;
    double flops = 2.0 * M * N * K;

    // ========== 预热 ==========
    std::cout << "Warming up...\n";
    for (int i = 0; i < warmup; ++i) {
        cublas.gemm(A.data(), B.data(), C.data(), M, N, K);
    }

    // ========== 含内存拷贝 ==========
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cublas.gemm(A.data(), B.data(), C.data(), M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gflops = flops / (elapsed * 1e6);
        std::cout << "[cuBLAS + memcpy] Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";
    }

    // ========== 仅 Kernel ==========
    {
        // 上传数据
        cublas.upload(A.data(), B.data(), M, N, K);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cublas.gemm_kernel_only(A.data(), B.data(), C.data(), M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gflops = flops / (elapsed * 1e6);
        std::cout << "[cuBLAS kernel]   Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";

        // 获取结果用于验证
        cublas.get_result(C.data(), M, N);
    }

    // ========== 正确性检查 ==========
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
    }
    std::cout << "\n[CHECK] Max diff vs Eigen: " << max_diff;
    if (max_diff < eps) {
        std::cout << " ✓ OK\n";
    } else {
        std::cout << " ✗ FAIL (eps=" << eps << ")\n";
    }

    return 0;
}
