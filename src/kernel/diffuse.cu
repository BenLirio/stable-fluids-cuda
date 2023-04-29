#include <util/macros.h>
#include <stdio.h>
#include <util/idx2.cuh>
#include <cuda_runtime.h>
#include <kernel/solve.cuh>
#include <gold/solve.cuh>
#include <util/state.h>

void kernel_diffuse_wrapper(state_t *state, float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  float divisor = 1 + 4*factor;
  if (OUTPUT&OUTPUT_SOLVE_ERROR) {
    float *expected_values;
    CUDA_CHECK(cudaMalloc(&expected_values, N*sizeof(float)));
    gold_solve_wrapper(expected_values, previous_values, values, factor, divisor);
    kernel_solve(state, previous_values, values, expected_values, factor, divisor, DIFFUSE_TAG);
    cudaFree(expected_values);
  } else {
    kernel_solve(state, previous_values, values, NULL, factor, divisor, DIFFUSE_TAG);
  }
}