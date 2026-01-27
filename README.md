# 矩阵乘法 (GEMM Benchmark)

本仓库用 **C++** 从零实现多种 GEMM（General Matrix–Matrix Multiply）算法，并在可控且公平的环境里与主流开源库（Eigen）对比。

目标不是"为了超越而超越"，而是学习优先：一步步解释为什么快/为什么慢，用真实的基准数据与可读代码帮助你理解优化背后的原理与取舍。

---

## 📊 性能对比结果

### 测试环境

| 项目 | 配置 |
|-----|------|
| **CPU** | Intel Core i7-13700 (8P+8E, 24 Threads) |
| **GPU** | NVIDIA GeForce RTX 4060 (24 SMs, SM 8.9, 8GB) |
| **内存** | DDR5 |
| **操作系统** | Linux (Ubuntu) |
| **编译器** | GCC 9.4.0 |
| **CUDA** | 12.4 |

### 性能对比 (1024 × 1024 × 1024)

| 排名 | 算法 | 时间 (ms) | 吞吐量 (GFLOP/s) | vs REF 加速比 | 正确性 |
|:---:|------|----------:|----------------:|-------------:|:------:|
| 🥇 | **CUDA (kernel only)** | 1.79 | **1197.4** | **127.8x** | ✅ |
| 🥈 | **CUDA (含内存拷贝)** | 3.03 | **709.2** | 75.6x | ✅ |
| 🥉 | **Eigen (多线程)** | 13.71 | **156.7** | 16.7x | ✅ |
| 4 | **AVX2 + Blocked** | 15.76 | **136.3** | 14.5x | ✅ |
| 5 | **Eigen (单线程 AVX)** | 17.13 | **125.4** | 13.4x | ✅ |
| 6 | **AVX2** | 19.94 | **107.7** | 11.5x | ✅ |
| 7 | **AVX2 + OpenMP** | 26.79 | **80.1** | 8.5x | ✅ |
| 8 | **OpenMP (标量)** | 44.13 | **48.7** | 5.2x | ✅ |
| 9 | **Blocked (标量)** | 111.33 | **19.3** | 2.1x | ✅ |
| 10 | **REF (基准)** | 229.10 | **9.4** | 1.0x | ✅ |

### 大矩阵测试 (2048 × 2048 × 2048)

| 算法 | 时间 (ms) | 吞吐量 (GFLOP/s) |
|------|----------:|----------------:|
| **CUDA (kernel)** | 14.08 | **1220.0** |
| **CUDA (含内存拷贝)** | 19.47 | **882.4** |
| **AVX2 + OpenMP** | 50.13 | **342.7** |
| **Eigen (多线程)** | 63.53 | **270.4** |
| **AVX2 + Blocked** | 134.39 | **127.8** |
| **AVX2** | 157.40 | **109.2** |

### 关键发现

1. 🚀 **CUDA 完胜** - GPU kernel 达到 **1197 GFLOP/s**，是 CPU 基准的 **127x 加速**
2. 📦 **AVX2 + 分块** 效果显著 - 达到 **136 GFLOP/s**，接近 Eigen 优化库水平
3. 🧱 **单纯分块** 在标量模式下提升有限（仅 2x）
4. ⚡ **AVX2 SIMD** 带来 **11x 加速**（107 vs 9.4 GFLOP/s）
5. 🧵 **多线程 OpenMP** 在无 SIMD 时效果有限，需配合向量化

---

## 🏗️ 实现的算法

| 后端 | 文件 | 优化技术 | 说明 |
|-----|------|---------|------|
| **REF** | `backend_ref.hpp` | 无 | 纯标量基准实现，禁用编译器向量化 |
| **Blocked** | `backend_blocked.hpp` | Cache Tiling | 3层循环分块，优化缓存局部性 |
| **AVX2** | `backend_avx2.hpp` | SIMD + Packing | AVX2/FMA 指令 + B矩阵打包 |
| **AVX2+Blocked** | `backend_avx2_blocked.hpp` | SIMD + 5层分块 | 6x16 micro-kernel + BLIS风格分块 |
| **OpenMP** | `backend_openmp.hpp` | 多线程 | OpenMP 并行化 |
| **AVX2+OpenMP** | `backend_avx2_openmp.hpp` | SIMD + 多线程 | AVX2 + OpenMP 混合并行 |
| **CUDA** | `backend_cuda.cuh` | GPU | Tiled 共享内存算法 |
| **NEON** | `backend_neon.hpp` | ARM SIMD | ARM 平台优化（x86自动回退到REF） |

---

## 依赖

- GCC 9.4.0+
- Eigen3
- OpenMP
- CUDA 11.0+ (可选，用于 GPU 加速)

---

## 构建与运行

### 快速开始

```bash
# 创建构建目录
mkdir build && cd build

# 配置（如有CUDA）
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..

# 或不带CUDA
cmake ..

# 编译
make -j$(nproc)
```

### 运行基准测试

```bash
# 使用脚本运行所有测试
./run_bench.sh

# 自定义矩阵大小
M=2048 N=2048 K=2048 ./run_bench.sh

# 单独运行某个后端
./build/bench_avx2 --m 1024 --n 1024 --k 1024 --iters 10
./build/bench_cuda --m 4096 --n 4096 --k 4096 --iters 5

# 参数说明
#   --m M        矩阵A的行数
#   --n N        矩阵B的列数
#   --k K        矩阵A的列数/B的行数
#   --iters R    测试迭代次数
#   --warmup W   预热次数
#   --threads T  线程数（OpenMP后端）
#   --seed S     随机种子
#   --eps E      误差阈值
```

### 可用的可执行文件

| 可执行文件 | 说明 |
|-----------|------|
| `bench_ref` | 标量参考实现 |
| `bench_blocked` | 分块算法 |
| `bench_avx2` | AVX2 SIMD |
| `bench_avx2_blocked` | AVX2 + 分块 |
| `bench_openmp` | OpenMP 多线程 |
| `bench_avx2_openmp` | AVX2 + OpenMP |
| `bench_cuda` | CUDA GPU |
| `bench_neon` | ARM NEON |

---

## 📁 项目结构

```
GEMM/
├── include/
│   ├── macro.hpp              # 通用宏定义
│   ├── backend_ref.hpp        # 标量参考实现
│   ├── backend_blocked.hpp    # 分块算法
│   ├── backend_avx2.hpp       # AVX2 SIMD
│   ├── backend_avx2_blocked.hpp  # AVX2 + 分块
│   ├── backend_openmp.hpp     # OpenMP 多线程
│   ├── backend_avx2_openmp.hpp   # AVX2 + OpenMP
│   ├── backend_cuda.cuh       # CUDA GPU
│   └── backend_neon.hpp       # ARM NEON
├── bench/
│   ├── bench_gemm_ref.cpp
│   ├── bench_gemm_blocked.cpp
│   ├── bench_gemm_avx2.cpp
│   ├── bench_gemm_avx2_blocked.cpp
│   ├── bench_gemm_openmp.cpp
│   ├── bench_gemm_avx2_openmp.cpp
│   ├── bench_gemm_cuda.cu
│   └── bench_gemm_neon.cpp
├── CMakeLists.txt
├── run_bench.sh               # 一键运行脚本
└── README.md
```

---

## 📚 学习资源

- [How to optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)

---

## License

MIT
