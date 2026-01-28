# 矩阵乘法优化全集 (GEMM Benchmark Suite)

本仓库用 **C++** 从零实现多种 GEMM（General Matrix–Matrix Multiply）算法，涵盖从标量到 GPU 的完整优化路径，与主流库（Eigen、cuBLAS）横向对比。

**目标**：学习优先 —— 一步步解释为什么快/为什么慢，用真实的基准数据与可读代码帮助你理解优化背后的原理与取舍。

---

## 📊 性能对比结果

### 测试环境

| 项目 | 配置 |
|-----|------|
| **CPU** | Intel Core i7-13700 (8P+8E, 24 Threads) |
| **GPU** | NVIDIA GeForce RTX 4060 (24 SMs, SM 8.9, 8GB GDDR6) |
| **内存** | DDR5 |
| **操作系统** | Linux (Ubuntu) |
| **编译器** | GCC 9.4.0 / NVCC 12.4 |
| **CUDA** | 12.4 |

### 🏆 性能排行榜 (2048 × 2048 × 2048)

| 排名 | 算法 | 时间 (ms) | 吞吐量 (GFLOP/s) | vs REF 加速比 | 正确性 |
|:---:|------|----------:|----------------:|-------------:|:------:|
| 🥇 | **cuBLAS (kernel)** | 2.01 | **8561.4** | **1002x** | ✅ |
| 🥈 | **CUDA v3 (优化kernel)** | 3.75 | **4583.9** | 537x | ✅ |
| 🥉 | **CUDA v2 (64x64 tile)** | 4.49 | **3828.4** | 448x | ✅ |
| 4 | **cuBLAS (含memcpy)** | 6.88 | **2497.5** | 292x | ✅ |
| 5 | **CUDA v3 (含memcpy)** | 8.70 | **1974.6** | 231x | ✅ |
| 6 | **CUDA v1 (kernel)** | 12.97 | **1324.2** | 155x | ✅ |
| 7 | **CUDA v1 (含memcpy)** | 18.41 | **933.0** | 109x | ✅ |
| 8 | **Eigen (多线程)** | 20.58 | **834.8** | 98x | ✅ |
| 9 | **AVX2 + OpenMP** | 46.64 | **368.4** | 43x | ✅ |
| 10 | **AVX2 + OMP + Blocked** | 111.06 | **154.7** | 18x | ✅ |
| 11 | **Eigen (单线程)** | 117.90 | **145.7** | 17x | ✅ |
| 12 | **AVX2 + Blocked** | 134.82 | **127.4** | 15x | ✅ |
| 13 | **AVX2** | 152.33 | **112.8** | 13x | ✅ |
| 14 | **OpenMP (标量)** | 172.14 | **99.8** | 12x | ✅ |
| 15 | **Strassen** | 876.33 | **19.6** | 2.3x | ✅ |
| 16 | **REF (基准)** | 2011.37 | **8.5** | 1.0x | ✅ |

### 关键发现

1. 🚀 **cuBLAS 遥遥领先** - kernel 达到 **8561 GFLOP/s**，是手写 CUDA 的 **6.5x**
2. 🎮 **手写 CUDA 优化有效** - v3 kernel 达到 **4584 GFLOP/s**，是 v1 的 **3.5x**
3. 📦 **AVX2 + Blocked** 达到 **127 GFLOP/s**，接近 Eigen 单线程水平
4. ⚡ **AVX2 + OpenMP** 组合 - **368 GFLOP/s**，多线程效果显著
5. 🔢 **Strassen** 在此规模下不具优势 - 递归开销抵消了算法复杂度优势
6. 🧵 **内存带宽瓶颈** - GPU 含 memcpy 时性能下降 40-70%

---

## 🏗️ 实现的算法

### CPU 后端

| 后端 | 文件 | 优化技术 | 说明 |
|-----|------|---------|------|
| **REF** | `backend_ref.hpp` | 无 | 纯标量基准，禁用编译器优化 |
| **Blocked** | `backend_blocked.hpp` | Cache Tiling | 3层循环分块，优化缓存局部性 |
| **AVX2** | `backend_avx2.hpp` | SIMD + Packing | AVX2/FMA 指令 + B矩阵打包 |
| **AVX2+Blocked** | `backend_avx2_blocked.hpp` | SIMD + 5层分块 | 6x16 micro-kernel + BLIS风格 |
| **OpenMP** | `backend_openmp.hpp` | 多线程 | OpenMP 并行化 |
| **AVX2+OpenMP** | `backend_avx2_openmp.hpp` | SIMD + 多线程 | 4x16 micro-kernel + OpenMP |
| **AVX2+OMP+Blocked** | `backend_avx2_openmp_blocked.hpp` | 三重优化 | 8x8 micro-kernel + 5层分块 + OpenMP |
| **Strassen** | `backend_strassen.hpp` | 分治算法 | O(n^2.807) 复杂度 |
| **NEON** | `backend_neon.hpp` | ARM SIMD | ARM 平台优化 |

### GPU 后端

| 后端 | 文件 | 优化技术 | 说明 |
|-----|------|---------|------|
| **CUDA v1** | `backend_cuda.cuh` | Tiled | 32x32 共享内存 tiling |
| **CUDA v2** | `backend_cuda_v2.cuh` | 寄存器 | 64x64 tile + 8x8 寄存器分块 |
| **CUDA v3** | `backend_cuda_v2.cuh` | 优化v3 | 64x64 tile + 4x4 thread tile |
| **cuBLAS** | `backend_cublas.cuh` | NVIDIA 官方库 | 理论峰值性能 |

---

## 依赖

- GCC 9.4.0+ (支持 C++14)
- Eigen3
- OpenMP
- CUDA 11.0+ (可选，GPU 加速)
- cuBLAS (可选，性能基准)

---

## 构建与运行

### 快速开始

```bash
# 克隆仓库
git clone <repo-url>
cd GEMM

# 创建构建目录
mkdir build && cd build

# 配置（带 CUDA）
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCUBLAS_LIBRARY=/usr/local/cuda/lib64/libcublas.so ..

# 或不带 CUDA
cmake ..

# 编译
make -j$(nproc)
```

### 运行基准测试

```bash
# 单独运行某个后端
./build/bench_avx2 --m 1024 --n 1024 --k 1024 --iters 10
./build/bench_cuda --m 2048 --n 2048 --k 2048 --iters 5
./build/bench_cublas --m 4096 --n 4096 --k 4096 --iters 5
./build/bench_avx2_openmp --m 1024 --n 1024 --k 1024 --threads 16

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

| 可执行文件 | 说明 | 主要优化 |
|-----------|------|---------|
| `bench_ref` | 标量参考 | 无优化基准 |
| `bench_blocked` | 分块算法 | Cache Tiling |
| `bench_avx2` | AVX2 SIMD | 向量化 |
| `bench_avx2_blocked` | AVX2 + 分块 | SIMD + Tiling |
| `bench_openmp` | OpenMP | 多线程 |
| `bench_avx2_openmp` | AVX2 + OpenMP | SIMD + 多线程 |
| `bench_avx2_openmp_blocked` | 三重优化 | SIMD + 多线程 + Tiling |
| `bench_strassen` | Strassen | 分治算法 |
| `bench_cuda` | CUDA v1 | GPU Tiled |
| `bench_cuda_v2` | CUDA v2/v3 | GPU 寄存器优化 |
| `bench_cublas` | cuBLAS | NVIDIA 官方库 |
| `bench_neon` | ARM NEON | ARM 平台 |

---

## 📁 项目结构

```
GEMM/
├── include/
│   ├── macro.hpp                   # 通用宏定义
│   ├── backend_ref.hpp             # 标量参考实现
│   ├── backend_blocked.hpp         # 分块算法
│   ├── backend_avx2.hpp            # AVX2 SIMD
│   ├── backend_avx2_blocked.hpp    # AVX2 + 分块
│   ├── backend_openmp.hpp          # OpenMP 多线程
│   ├── backend_avx2_openmp.hpp     # AVX2 + OpenMP
│   ├── backend_avx2_openmp_blocked.hpp  # 三重优化
│   ├── backend_strassen.hpp        # Strassen 算法
│   ├── backend_cuda.cuh            # CUDA GPU v1
│   ├── backend_cuda_v2.cuh         # CUDA GPU v2/v3
│   ├── backend_cublas.cuh          # cuBLAS
│   └── backend_neon.hpp            # ARM NEON
├── bench/
│   ├── bench_gemm_*.cpp            # CPU 基准测试
│   └── bench_gemm_*.cu             # GPU 基准测试
├── .github/workflows/
│   └── c-cpp.yml                   # CI/CD 配置
├── CMakeLists.txt
├── run_bench.sh                    # 一键运行脚本
└── README.md
```

---

## 📈 优化路线图

```
REF (8.5 GFLOP/s)
  │
  ├─> Blocked (21 GFLOP/s, 2.5x) ─┐
  │                               │
  └─> AVX2 (113 GFLOP/s, 13x) ────┼─> AVX2+Blocked (127 GFLOP/s, 15x)
                                  │
  OpenMP (100 GFLOP/s, 12x) ──────┼─> AVX2+OpenMP (368 GFLOP/s, 43x)
                                  │
                                  └─> AVX2+OMP+Blocked (155 GFLOP/s, 18x)

GPU:
  CUDA v1 (1324 GFLOP/s) ─> CUDA v3 (4584 GFLOP/s) ─> cuBLAS (8561 GFLOP/s)
```

---

## 📚 学习资源

- [How to optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Library Documentation](https://docs.nvidia.com/cuda/cublas/)

---

## License

MIT
