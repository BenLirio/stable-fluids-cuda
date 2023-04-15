#ifndef STABLE_FLUIDS_CUDA_KERNEL_ADVECT_H_
#define STABLE_FLUIDS_CUDA_KERNEL_ADVECT_H_

void kernel_advect_wrapper(float *previous_values, float *values, float *x_velocities, float *y_velocities);

#endif // STABLE_FLUIDS_CUDA_KERNEL_ADVECT_H_