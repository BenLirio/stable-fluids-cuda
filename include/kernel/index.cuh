#ifndef STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_
#define STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_
#include <util/state.h>

void kernel_step(state_t state, int current_step);
void kernel_step_wrapper(state_t state, int current_step);

#endif // STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_