#ifndef STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_
#define STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_

#include <cuda_runtime.h>

void kernel_project_test_harness(float *x_velocities, float *y_velocities, float *pressure, float *divergence);
void kernel_project_wrapper(float *x_velocities, float *y_velocities, float *pressure, float *divergence);

extern void (*kernel_project)(float *x_velocities, float *y_velocities, float *pressure, float *divergence);
extern void (*kernel_project_solve_red_black)(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red);

__global__ void kernel_project_solve_red_black_naive(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red);
__global__ void kernel_project_solve_red_black_shared(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red);

#endif // STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_