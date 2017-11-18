#ifndef CUDA_FUNC_H_
#define CUDA_FUNC_H_

void gpu_mem_alloc(float **x, int64_t n);
void gpu_init(int64_t n, float *x, float val);
void gpu_reduce(int64_t n, float *x, float *y);

#endif/* cuda_func.h */
