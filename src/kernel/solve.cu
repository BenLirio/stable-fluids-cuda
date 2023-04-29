#include <kernel/solve.cuh>
#include <util/macros.h>
#include "cuda_runtime.h"
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/performance.cuh>
#include <stdlib.h>
#include <unistd.h>

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

__global__ void kernel_solve_no_block_sync(float *base, float *values, float factor, float divisor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);
  float next_value = (base[IDX2(idx)] +
    factor*(
      values[IDX2(idx2_add(idx, idx2(1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(0, 1)))] +
      values[IDX2(idx2_add(idx, idx2(0, -1)))]
    )) / divisor;
  __syncthreads();
  values[IDX2(idx)] = next_value;
}


float kernel_solve_get_error(float *values, float *expected_values) {
  float *host_values = (float*)malloc(N*sizeof(float));
  cudaMemcpy(host_values, values, N*sizeof(float), cudaMemcpyDeviceToHost);
  float *host_expected_values = (float*)malloc(N*sizeof(float));
  cudaMemcpy(host_expected_values, expected_values, N*sizeof(float), cudaMemcpyDeviceToHost);
  float error = 0.0f;
  for (int i = 0; i < N; i++) {
    error += fabsf(host_values[i] - host_expected_values[i]);
  }
  free(host_values);
  free(host_expected_values);
  return error / N;
  // float error = 0.0f;
  // float *device_error;
  // cudaMalloc(&device_error, sizeof(float));
  // cudaMemcpy(device_error, &error, sizeof(float), cudaMemcpyHostToDevice);
  // kernel_solve_sum_error<<<GRID_DIM, BLOCK_DIM>>>(device_error, values, expected_values);
  // KERNEL_ERROR_CHECK();
  // cudaMemcpy(&error, device_error, sizeof(float), cudaMemcpyDeviceToHost);
  // cudaFree(device_error);
  // error /= N;
  // return error;
}

int kernel_solve(int step, float *base, float *values, float *expected_values, float factor, float divisor, int tags) {
  for (int num_iterations = 1; num_iterations <= MAX_CONVERGENCE_ITERATIONS; num_iterations++) {
    // for (int red_black = 0; red_black < 2; red_black++)
    //   kernel_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(red_black, base, values, factor, divisor);
    kernel_solve_no_block_sync<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor);

    if (expected_values != NULL) {
      float error = kernel_solve_get_error(values, expected_values);

      if (OUTPUT_PERFORMANCE) {
        print_tags(tags|SOLVE_TAG);
        printf("[step=%d][gauss_step=%d][error=%f]\n", step, num_iterations, error);
      }

      if (error < EQ_THRESHOLD) return num_iterations;

    } else {
      if (num_iterations == GAUSS_SEIDEL_ITERATIONS) return GAUSS_SEIDEL_ITERATIONS;
    }
  }
  return MAX_CONVERGENCE_ITERATIONS;
}

// __global__ void kernel_solve_sum_error(float *error, float *values, float *expected_values) {
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int xoff = blockIdx.x * blockDim.x;
//   int yoff = blockIdx.y * blockDim.y;
//   int x = xoff + tx + 1;
//   int y = yoff + ty + 1;
//   if (x > WIDTH || y > HEIGHT) return;
//   idx2 idx = idx2(x, y);
//   __shared__ float shared_error[BLOCK_SIZE][BLOCK_SIZE];
//   shared_error[ty][tx] = fabsf(values[IDX2(idx)] - expected_values[IDX2(idx)]);
//   __syncthreads();
//   if (tx == 0) {
//     for (int i = 1; i < BLOCK_SIZE; i++) {
//       if (xoff + i + 1 > WIDTH) break;
//       shared_error[ty][0] += shared_error[ty][i];
//     }
//   }
//   __syncthreads();
//   if (tx == 0 && ty == 0) {
//     for (int i = 1; i < BLOCK_SIZE; i++) {
//       if (yoff + i + 1 > HEIGHT) break;
//       shared_error[0][0] += shared_error[i][0];
//     }
//     if (shared_error[0][0] != shared_error[0][0])
//     atomicAdd(error, shared_error[0][0]);
//   }

// }