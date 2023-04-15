#ifndef STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_
#define STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_

void kernel_diffuse(float *previous_values, float *values, float rate);
void kernel_diffuse_wrapper(float *previous_values, float *values, float rate);

#endif // STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_