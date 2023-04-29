#ifndef STABLE_FLUIDS_CUDA_SOLVE_H_
#define STABLE_FLUIDS_CUDA_SOLVE_H_

#include <util/state.h>

int kernel_solve(state_t *state, float *base, float *values, float *expected_values, float factor, float divisor, int tags);

#endif // STABLE_FLUIDS_CUDA_SOLVE_H_