// Copyright 2024-2026 Auaoalg
// SPDX-License-Identifier: MIT
//
// AVX2 + OpenMP + Blocked GEMM Benchmark (Triple Optimization)

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
#include "macro.hpp"
#include "backend_avx2_openmp_blocked.hpp"

using namespace Auaoalg;

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int iters = 10;
    int warmup = 3;
    int threads = omp_get_max_threads();
    unsigned seed = 42;
    float eps = 1e-3f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = atof(argv[++i]);
    }

    omp_set_num_threads(threads);

    std::cout << "========================================\n";
    std::cout << "AVX2 + OpenMP + Blocked GEMM Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix size: " << M << " x " << K << " x " << N << "\n";
    std::cout << "Threads: " << threads << "\n";
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

    double flops = 2.0 * M * N * K;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        gemm_avx2_openmp_blocked(A.data(), B.data(), C.data(), M, N, K);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        gemm_avx2_openmp_blocked(A.data(), B.data(), C.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gflops = flops / (elapsed * 1e6);

    std::cout << "[AVX2+OMP+Blocked] Time: " << elapsed << " ms, " << gflops << " GFLOP/s\n";

    // 正确性检查
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
