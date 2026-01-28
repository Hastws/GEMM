#!/bin/bash

# GEMM Benchmark 对比脚本
# 运行所有后端的基准测试，并对比结果

set -e

# 默认参数
M=${M:-1024}
N=${N:-1024}
K=${K:-1024}
ITERS=${ITERS:-5}
WARMUP=${WARMUP:-2}
THREADS=${THREADS:-0}  # 0 means auto
SEED=${SEED:-42}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

BUILD_DIR="build"

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       ${CYAN}GEMM Benchmark Comparison Tool${BLUE}                           ║${NC}"
    echo -e "${BLUE}║       ${NC}Compare different matrix multiplication algorithms${BLUE}        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_header
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Matrix Size: M=${M} N=${N} K=${K}"
echo "  Iterations: ${ITERS}  Warmup: ${WARMUP}"
echo "  FLOPs per run: $(echo "scale=2; 2 * $M * $N * $K / 1000000000" | bc) GFLOP"
echo ""

# 检查构建目录是否存在
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Build directory not found, creating...${NC}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    export PATH=/usr/local/cuda/bin:$PATH
    cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .. > /dev/null 2>&1 || cmake .. > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1
    cd ..
fi

cd "$BUILD_DIR"

# 运行参数
ARGS="--m $M --n $N --k $K --iters $ITERS --warmup $WARMUP --seed $SEED"
ARGS_MT="$ARGS --threads $THREADS"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    Running Benchmarks${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

declare -A RESULTS

# 运行 REF 后端
if [ -f "bench_ref" ]; then
    echo -e "${YELLOW}[1/7] REF (Scalar, No Vectorization)${NC}"
    echo "────────────────────────────────────────"
    ./bench_ref $ARGS
    echo ""
fi

# 运行 Blocked 后端
if [ -f "bench_blocked" ]; then
    echo -e "${YELLOW}[2/7] Blocked (Cache Tiling)${NC}"
    echo "────────────────────────────────────────"
    ./bench_blocked $ARGS
    echo ""
fi

# 运行 AVX2 后端
if [ -f "bench_avx2" ]; then
    echo -e "${YELLOW}[3/7] AVX2 (SIMD Vectorization)${NC}"
    echo "────────────────────────────────────────"
    ./bench_avx2 $ARGS
    echo ""
fi

# 运行 AVX2+Blocked 后端
if [ -f "bench_avx2_blocked" ]; then
    echo -e "${YELLOW}[4/7] AVX2 + Blocked (SIMD + Cache Tiling)${NC}"
    echo "────────────────────────────────────────"
    ./bench_avx2_blocked $ARGS
    echo ""
fi

# 运行 OpenMP 后端
if [ -f "bench_openmp" ]; then
    echo -e "${YELLOW}[5/7] OpenMP (Multi-threaded)${NC}"
    echo "────────────────────────────────────────"
    ./bench_openmp $ARGS_MT
    echo ""
fi

# 运行 AVX2+OpenMP 后端
if [ -f "bench_avx2_openmp" ]; then
    echo -e "${YELLOW}[6/7] AVX2 + OpenMP (SIMD + Multi-threaded)${NC}"
    echo "────────────────────────────────────────"
    ./bench_avx2_openmp $ARGS_MT
    echo ""
fi

# 运行 CUDA 后端
if [ -f "bench_cuda" ]; then
    echo -e "${YELLOW}[7/7] CUDA (GPU Acceleration)${NC}"
    echo "────────────────────────────────────────"
    ./bench_cuda $ARGS
    echo ""
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Benchmark Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Tips:"
echo "  - Use environment variables to customize: M=2048 N=2048 K=2048 ./run_bench.sh"
echo "  - All results show [CHECK] OK if computation is correct"
echo "  - GFLOP/s = 2*M*N*K / (time_ms * 1e6)"
