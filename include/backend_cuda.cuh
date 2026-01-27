// CUDA GEMM Implementation
// Requires CUDA 11.0+ and compute capability 3.5+

#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_CUDA_CUH
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_BACKEND_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdio>

namespace Auaoalg {
namespace CUDA {

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Naive CUDA kernel - one thread per output element
__global__ void GemmNaiveKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled/Shared memory CUDA kernel - uses shared memory for blocking
template<int TILE_SIZE>
__global__ void GemmTiledKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized tiled kernel with more work per thread
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void GemmOptimizedKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Each thread computes a 4x4 block
    const int THREAD_M = 4;
    const int THREAD_N = 4;
    
    int row_start = by * TILE_M + ty * THREAD_M;
    int col_start = bx * TILE_N + tx * THREAD_N;
    
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Load A tile (each thread loads multiple elements)
        #pragma unroll
        for (int i = 0; i < THREAD_M; ++i) {
            int global_row = row_start + i;
            int global_col = t * TILE_K + tx;
            if (global_row < M && global_col < K) {
                As[tx][ty * THREAD_M + i] = A[global_row * K + global_col];
            } else {
                As[tx][ty * THREAD_M + i] = 0.0f;
            }
        }
        
        // Load B tile
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            int global_row = t * TILE_K + ty;
            int global_col = col_start + j;
            if (global_row < K && global_col < N) {
                Bs[ty][tx * THREAD_N + j] = B[global_row * N + global_col];
            } else {
                Bs[ty][tx * THREAD_N + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                float a_val = As[k][ty * THREAD_M + i];
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += a_val * Bs[k][tx * THREAD_N + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        int global_row = row_start + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                int global_col = col_start + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = acc[i][j];
                }
            }
        }
    }
}

// Device memory manager
class GemmCUDA {
public:
    GemmCUDA() : d_A(nullptr), d_B(nullptr), d_C(nullptr), 
                 allocated_M(0), allocated_N(0), allocated_K(0) {}
    
    ~GemmCUDA() {
        Free();
    }
    
    void Free() {
        if (d_A) { cudaFree(d_A); d_A = nullptr; }
        if (d_B) { cudaFree(d_B); d_B = nullptr; }
        if (d_C) { cudaFree(d_C); d_C = nullptr; }
        allocated_M = allocated_N = allocated_K = 0;
    }
    
    void Allocate(int M, int N, int K) {
        if (M <= allocated_M && N <= allocated_N && K <= allocated_K) {
            return;  // Already have enough memory
        }
        Free();
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        allocated_M = M;
        allocated_N = N;
        allocated_K = K;
    }
    
    // Naive GEMM
    void MatrixMultiplyNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
        Allocate(M, N, K);
        
        CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        
        GemmNaiveKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Tiled GEMM with shared memory
    void MatrixMultiplyTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
        Allocate(M, N, K);
        
        CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        
        constexpr int TILE_SIZE = 32;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        GemmTiledKernel<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Kernel only (for benchmarking without memory transfer)
    void MatrixMultiplyTiledKernelOnly(int M, int N, int K) {
        constexpr int TILE_SIZE = 32;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        GemmTiledKernel<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Upload data (for kernel-only benchmarking)
    void UploadData(const float* A, const float* B, int M, int N, int K) {
        Allocate(M, N, K);
        CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void DownloadResult(float* C, int M, int N) {
        CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
private:
    float *d_A, *d_B, *d_C;
    int allocated_M, allocated_N, allocated_K;
};

}  // namespace CUDA
}  // namespace Auaoalg

#endif
