#ifndef STABLE_FLUIDS_CUDA_SOURCE_VELOCITIES_H_
#define STABLE_FLUIDS_CUDA_SOURCE_VELOCITIES_H_

#include <cuda_runtime.h>

__global__ void kernel_source_velocities(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step);
void kernel_source_velocities_wrapper(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step);

#endif //STABLE_FLUIDS_CUDA_SOURCE_VELOCITIES_H_