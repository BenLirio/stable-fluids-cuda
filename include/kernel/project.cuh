#ifndef STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_
#define STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_

#include <cuda_runtime.h>

__global__ void kernel_project(float *x_velocities, float *y_velocities, float *pressure, float *divergence);
void kernel_project_wrapper(float *x_velocities, float *y_velocities, float *pressure, float *divergence);

#endif // STABLE_FLUIDS_CUDA_KERNEL_PROJECT_H_