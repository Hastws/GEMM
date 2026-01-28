// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// Strassen Algorithm GEMM Benchmark

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "macro.hpp"
#include "backend_strassen.hpp"

using namespace Auaoalg;

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int iters = 5;  // Strassen 较慢，减少迭代
    int warmup = 2;
    unsigned seed = 42;
    float eps = 1e-2f;  // Strassen 数值精度稍差

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
    std::cout << "Strassen Algorithm GEMM Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix size: " << M << " x " << K << " x " << N << "\n";
    std::cout << "Iterations: " << iters << ", Warmup: " << warmup << "\n";
    std::cout << "Complexity: O(n^2.807) vs O(n^3)\n\n";

    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // Eigen 参考
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eA(A.data(), M, K), eB(B.data(), K, N), eC(C_ref.data(), M, N);
    eC = eA * eB;

    double flops = 2.0 * M * N * K;  // 等效 FLOPS（用于对比）

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < warmup; ++i) {
        gemm_strassen(A.data(), B.data(), C.data(), M, N, K);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        gemm_strassen(A.data(), B.data(), C.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gflops = flops / (elapsed * 1e6);

    std::cout << "[Strassen] Time: " << elapsed << " ms, " << gflops << " GFLOP/s (equivalent)\n";

    // 正确性检查
    float max_diff = 0.0f;
    double rel_err_sum = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(C[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);
        if (std::abs(C_ref[i]) > 1e-6f) {
            rel_err_sum += diff / std::abs(C_ref[i]);
        }
    }
    double avg_rel_err = rel_err_sum / (M * N);

    std::cout << "\n[CHECK] Max diff vs Eigen: " << max_diff << "\n";
    std::cout << "[CHECK] Avg relative error: " << avg_rel_err << "\n";
    if (max_diff < eps) {
        std::cout << "[CHECK] ✓ OK\n";
    } else {
        std::cout << "[CHECK] ✗ FAIL (eps=" << eps << ")\n";
        std::cout << "Note: Strassen has inherent numerical instability\n";
    }

    return 0;
}
