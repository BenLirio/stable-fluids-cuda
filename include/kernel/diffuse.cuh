#ifndef STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_
#define STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_

#include <cuda_runtime.h>

void kernel_diffuse_test_harness(float *previous_values, float *values, float rate);
void kernel_diffuse_wrapper(float *previous_values, float *values, float rate);

extern void (*kernel_diffuse)(float *previous_values, float *values, float rate);
extern void (*kernel_diffuse_red_black)(float *previous_values, float *values, float rate, int red);


__global__ void kernel_diffuse_red_black_shared(float *previous_values, float *values, float rate, int red);
__global__ void kernel_diffuse_red_black_naive(float *previous_values, float *values, float rate, int red);

#endif // STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_