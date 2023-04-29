#ifndef STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_
#define STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_

#include <cuda_runtime.h>
#include <util/state.h>

void kernel_diffuse_wrapper(state_t *state, float *previous_values, float *values, float rate);

#endif // STABLE_FLUIDS_CUDA_KERNEL_DIFFUSE_H_