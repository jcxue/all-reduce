#include "cuda_func.h"

__global__ void init_kernel(int64_t n, float *x, float val)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = index; i < n; i += stride) {
    	x[i] = val;
    }
}

__global__ void reduce_kernel(int64_t n, float *x, float *y)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = index; i < n; i += stride) {
    	y[i] = x[i] + y[i];
    }
}

void gpu_mem_alloc(float **x, int64_t n)
{
    cudaMallocManaged(x, n*sizeof(float));
    cudaDeviceSynchronize();
}

void gpu_init(int64_t n, float *x, float val)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    init_kernel<<<numBlocks, blockSize>>>(n, x, val);
    cudaDeviceSynchronize();
}

void gpu_reduce(int64_t n, float *x, float *y)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    reduce_kernel<<<numBlocks, blockSize>>>(n, x, y);
    cudaDeviceSynchronize();
}