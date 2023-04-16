#ifndef STABLE_FLUIDS_CUDA_UTIL_DERIVATIVE_H_
#define STABLE_FLUIDS_CUDA_UTIL_DERIVATIVE_H_

#include <cuda_runtime.h>
#include <util/idx2.cuh>

__device__ __host__ float get_x_derivative(float *x_velocities, idx2 idx);
__device__ __host__ float get_y_derivative(float *y_velocities, idx2 idx);

#endif // STABLE_FLUIDS_CUDA_UTIL_DERIVATIVE_H_
