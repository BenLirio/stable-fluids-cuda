#ifndef STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_
#define STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_

int gold_solve(float *b, float *x, float factor, float divisor, float *expected_x, bool print_performance);

#endif // STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_