#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "backend_blocked.hpp"

using Clock = std::chrono::high_resolution_clock;

struct Args {
  int M = 512, N = 1024, K = 64;
  int iters = 10;
  int warmup = 2;
  int threads = 1;
  std::uint32_t seed = 42;
  float check_eps = 1e-3f;
};

static void ParseInt(const char* s, int& out) { out = std::stoi(s); }
static void ParseU32(const char* s, std::uint32_t& out) {
  out = static_cast<std::uint32_t>(std::stoul(s));
}
static void ParseFloat(const char* s, float& out) { out = std::stof(s); }

static Args ParseArgs(const int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](int i) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value after " << k << std::endl;
        std::exit(1);
      }
    };
    if (k == "--m") { need(i); ParseInt(argv[++i], a.M); }
    else if (k == "--n") { need(i); ParseInt(argv[++i], a.N); }
    else if (k == "--k") { need(i); ParseInt(argv[++i], a.K); }
    else if (k == "--iters") { need(i); ParseInt(argv[++i], a.iters); }
    else if (k == "--warmup") { need(i); ParseInt(argv[++i], a.warmup); }
    else if (k == "--threads") { need(i); ParseInt(argv[++i], a.threads); }
    else if (k == "--seed") { need(i); ParseU32(argv[++i], a.seed); }
    else if (k == "--eps") { need(i); ParseFloat(argv[++i], a.check_eps); }
    else if (k == "-h" || k == "--help") {
      std::cout << "Usage: bench_blocked [--m M] [--n N] [--k K] [--iters R] "
                   "[--warmup W] [--seed S] [--eps E]\n";
      std::exit(0);
    }
  }
  return a;
}

static double Gflops(int M, int N, int K, double ms) {
  double flops = 2.0 * M * N * K;
  return flops / (ms * 1e6);
}

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  const int M = args.M, N = args.N, K = args.K;

  std::vector<float> A(M * K), B(K * N), C_user(M * N), C_eigen(M * N);

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);

  Eigen::setNbThreads(args.threads);
  using MatRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const MatRM> EA(A.data(), M, K);
  Eigen::Map<const MatRM> EB(B.data(), K, N);
  Eigen::Map<MatRM> EC(C_eigen.data(), M, N);

  EC.setZero();
  EC.noalias() = EA * EB;

  std::fill(C_user.begin(), C_user.end(), 0.f);
  Auaoalg::Blocked::MatrixMultiply(A.data(), B.data(), C_user.data(), M, N, K);

  float max_abs = 0.f, max_rel = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float ref = C_eigen[i], got = C_user[i];
    float da = std::abs(got - ref);
    float dr = da / std::max(std::abs(ref), 1e-12f);
    max_abs = std::max(max_abs, da);
    max_rel = std::max(max_rel, dr);
  }
  bool ok = (max_abs <= args.check_eps) || (max_rel <= 1e-4f);
  if (!ok) {
    std::cerr << "[CHECK] FAILED: max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    return 1;
  }
  std::cout << "[CHECK] OK: max_abs=" << max_abs << " max_rel=" << max_rel << "\n";

  for (int w = 0; w < args.warmup; ++w) {
    std::fill(C_user.begin(), C_user.end(), 0.f);
    Auaoalg::Blocked::MatrixMultiply(A.data(), B.data(), C_user.data(), M, N, K);
  }

  double best_eigen_ms = 1e100, best_user_ms = 1e100;
  for (int r = 0; r < args.iters; ++r) {
    EC.setZero();
    auto t0 = Clock::now();
    EC.noalias() = EA * EB;
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    best_eigen_ms = std::min(best_eigen_ms, ms);
  }
  for (int r = 0; r < args.iters; ++r) {
    std::fill(C_user.begin(), C_user.end(), 0.f);
    auto t0 = Clock::now();
    Auaoalg::Blocked::MatrixMultiply(A.data(), B.data(), C_user.data(), M, N, K);
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    best_user_ms = std::min(best_user_ms, ms);
  }

  std::cout.setf(std::ios::fixed);
  std::cout.precision(3);
  std::cout << "M=" << M << " N=" << N << " K=" << K << " | iters=" << args.iters << "\n";
  std::cout << "Eigen      : " << best_eigen_ms << " ms | " << Gflops(M, N, K, best_eigen_ms) << " GFLOP/s\n";
  std::cout << "Blocked    : " << best_user_ms << " ms | " << Gflops(M, N, K, best_user_ms) << " GFLOP/s\n";
  return 0;
}
