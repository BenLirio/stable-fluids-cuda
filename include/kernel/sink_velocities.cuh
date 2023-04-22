#ifndef STABLE_FLUIDS_CUDA_SINK_VELOCITIES_H_
#define STABLE_FLUIDS_CUDA_SINK_VELOCITIES_H_
#include <cuda_runtime.h>

extern void (*kernel_sink_velocities)(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities);
void kernel_sink_velocities_wrapper(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities);

#endif //STABLE_FLUIDS_CUDA_SINK_VELOCITIES_H_