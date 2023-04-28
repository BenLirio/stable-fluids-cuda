#include <kernel/solve.cuh>
#include <util/macros.h>
#include "cuda_runtime.h"
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/performance.cuh>

__global__ void kernel_solve_red_black(int red_black, float *base, float *values, float factor, float divisor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);

  if ((x%2) == ((y+red_black)%2)) {
    values[IDX2(idx)] = (base[IDX2(idx)] +
      factor*(
        values[IDX2(idx2_add(idx, idx2(1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(0, 1)))] +
        values[IDX2(idx2_add(idx, idx2(0, -1)))]
      )) / divisor;
  }
}

__global__ void kernel_solve_sum_error(float *error, float *values, float *expected_values) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);
  __shared__ float shared_error[BLOCK_SIZE][BLOCK_SIZE];
  shared_error[ty][tx] = fabsf(values[IDX2(idx)] - expected_values[IDX2(idx)]);

  __syncthreads();
  if (tx == 0) {
    for (int i = 1; i < BLOCK_SIZE; i++) {
      shared_error[ty][0] += shared_error[ty][i];
    }
  }
  __syncthreads();
  if (tx == 0 && ty == 0) {
    for (int i = 1; i < BLOCK_SIZE; i++) {
      shared_error[0][0] += shared_error[i][0];
    }
    atomicAdd(error, shared_error[0][0]);
  }

}

int kernel_solve(float *base, float *values, float *expected_values, float factor, float divisor) {
  float error;
  float *device_error;
  if (expected_values != NULL)
    cudaMalloc(&device_error, sizeof(float));

  for (int num_iterations = 1;; num_iterations++) {
    for (int red_black = 0; red_black < 2; red_black++)
      kernel_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(red_black, base, values, factor, divisor);

    if (expected_values != NULL) {
      error = 0.0f;
      cudaMemcpy(device_error, &error, sizeof(float), cudaMemcpyHostToDevice);
      kernel_solve_sum_error<<<GRID_DIM, BLOCK_DIM>>>(device_error, values, expected_values);
      cudaMemcpy(&error, device_error, sizeof(float), cudaMemcpyDeviceToHost);
      error /= N;
      print_tags(SOLVE_TAG);
      if (OUTPUT_PERFORMANCE)
        printf("[step=%d][error=%f]\n", num_iterations, error);
      if (error < GUASS_SEIDEL_THREASHOLD || num_iterations == MAX_CONVERGENCE_ITERATIONS) {
        cudaFree(device_error);
        return num_iterations;
      }
    } else {
      if (num_iterations > GAUSS_SEIDEL_ITERATIONS) {
        return num_iterations;
      }
    }
  }
}