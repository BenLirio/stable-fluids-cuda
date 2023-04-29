#include <kernel/solve.cuh>
#include <util/macros.h>
#include "cuda_runtime.h"
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/performance.cuh>
#include <stdlib.h>
#include <unistd.h>

__global__ void kernel_solve_no_block_sync(float *base, float *values, float factor, float divisor, int iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);
  for (int i = 0; i < iterations; i++) {
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
}

__global__ void kernel_solve_thread_fence(float *base, float *values, float factor, float divisor, int iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);
  for (int i = 0; i < iterations; i++) {
    float next_value = (base[IDX2(idx)] +
      factor*(
        values[IDX2(idx2_add(idx, idx2(1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(0, 1)))] +
        values[IDX2(idx2_add(idx, idx2(0, -1)))]
      )) / divisor;
    __syncthreads();
    values[IDX2(idx)] = next_value;
    __threadfence();
  }
}

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

__global__ void kernel_solve_red_black_thread_fence(float *base, float *values, float factor, float divisor, int iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);
  for (int i = 0; i < iterations; i++) {
    for (int red_black = 0; red_black < 2; red_black++) {
      if ((x%2) == ((y+red_black)%2)) {
        values[IDX2(idx)] = (base[IDX2(idx)] +
          factor*(
            values[IDX2(idx2_add(idx, idx2(1, 0)))] +
            values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
            values[IDX2(idx2_add(idx, idx2(0, 1)))] +
            values[IDX2(idx2_add(idx, idx2(0, -1)))]
          )) / divisor;
      }
      __threadfence();
    }
  }
}

__global__ void kernel_solve_thread_fence_shared_memory(float *base, float *values, float factor, float divisor, int iterations) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int offx = blockIdx.x * blockDim.x;
  int offy = blockIdx.y * blockDim.y;
  int x = offx + tx + 1;
  int y = offy + ty + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);

  __shared__ float shared_values[BLOCK_SIZE][BLOCK_SIZE];
  float base_register = base[IDX2(idx)];
  shared_values[ty][tx] = values[IDX2(idx)];
  __syncthreads();

  for (int i = 0; i < iterations; i++) {
    float next_value = base_register;
    next_value += tx==0
      ? factor*values[IDX2(idx2_add(idx, idx2(-1, 0)))]
      : factor*shared_values[ty][tx-1];
    next_value += tx==BLOCK_SIZE-1 || x==WIDTH
      ? factor*values[IDX2(idx2_add(idx, idx2(1, 0)))]
      : factor*shared_values[ty][tx+1];
    next_value += ty==0
      ? factor*values[IDX2(idx2_add(idx, idx2(0, -1)))]
      : factor*shared_values[ty-1][tx];
    next_value += ty==BLOCK_SIZE-1 || y==HEIGHT
      ? factor*values[IDX2(idx2_add(idx, idx2(0, 1)))]
      : factor*shared_values[ty+1][tx];
    next_value /= divisor;
    if (tx==0||tx==BLOCK_SIZE-1||x==WIDTH||ty==0||ty==BLOCK_SIZE-1||y==HEIGHT)
      values[IDX2(idx)] = next_value;
    __threadfence();
    __syncthreads();
    shared_values[ty][tx] = next_value;
    __syncthreads();
  }
  values[IDX2(idx)] = shared_values[ty][tx];
}

__global__ void kernel_solve_red_black_thread_fence_shared_memory(float *base, float *values, float factor, float divisor, int iterations) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int offx = blockIdx.x * blockDim.x;
  int offy = blockIdx.y * blockDim.y;
  int x = offx + tx + 1;
  int y = offy + ty + 1;
  if (x > WIDTH || y > HEIGHT) return;
  idx2 idx = idx2(x, y);

  __shared__ float shared_values[BLOCK_SIZE][BLOCK_SIZE];
  float base_register = base[IDX2(idx)];
  shared_values[ty][tx] = values[IDX2(idx)];
  __syncthreads();

  for (int i = 0; i < iterations; i++) {
    for (int red_black = 0; red_black < 2; red_black++) {
      if ((x%2) == ((y+red_black)%2)) {
        shared_values[ty][tx] = base_register;
        shared_values[ty][tx] += tx==0
          ? factor*values[IDX2(idx2_add(idx, idx2(-1, 0)))]
          : factor*shared_values[ty][tx-1];
        shared_values[ty][tx] += tx==BLOCK_SIZE-1 || x==WIDTH
          ? factor*values[IDX2(idx2_add(idx, idx2(1, 0)))]
          : factor*shared_values[ty][tx+1];
        shared_values[ty][tx] += ty==0
          ? factor*values[IDX2(idx2_add(idx, idx2(0, -1)))]
          : factor*shared_values[ty-1][tx];
        shared_values[ty][tx] += ty==BLOCK_SIZE-1 || y==HEIGHT
          ? factor*values[IDX2(idx2_add(idx, idx2(0, 1)))]
          : factor*shared_values[ty+1][tx];
        shared_values[ty][tx] /= divisor;
        if (tx==0||tx==BLOCK_SIZE-1||x==WIDTH||ty==0||ty==BLOCK_SIZE-1||y==HEIGHT)
          values[IDX2(idx)] = shared_values[ty][tx];
      }
      __syncthreads();
      __threadfence();
    }
  }
  values[IDX2(idx)] = shared_values[ty][tx];
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
}

bool is_kernel_iterative() {
  return (
    (KERNEL_FLAGS==USE_NO_BLOCK_SYNC) ||
    (KERNEL_FLAGS==USE_THREAD_FENCE) ||
    (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_RED_BLACK)) ||
    (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_RED_BLACK|USE_SHARED_MEMORY)) ||
    (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_SHARED_MEMORY))
  );
}
bool is_host_iterative() {
  return (
    (KERNEL_FLAGS==USE_RED_BLACK)
  );
}

int kernel_solve(int step, float *base, float *values, float *expected_values, float factor, float divisor, int tags) {
  if (expected_values != NULL) {
    if (is_kernel_iterative()) {
      int num_iterations;
      float *saved_base, *saved_values;
      cudaMalloc(&saved_base, N*sizeof(float));
      cudaMalloc(&saved_values, N*sizeof(float));
      cudaMemcpy(saved_base, base, N*sizeof(float), cudaMemcpyDeviceToDevice);
      cudaMemcpy(saved_values, values, N*sizeof(float), cudaMemcpyDeviceToDevice);
      for (num_iterations = 1; num_iterations <= MAX_CONVERGENCE_ITERATIONS; num_iterations++) {
        cudaMemcpy(base, saved_base, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(values, saved_values, N*sizeof(float), cudaMemcpyDeviceToDevice);
        if (KERNEL_FLAGS==USE_NO_BLOCK_SYNC)
          kernel_solve_no_block_sync<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, num_iterations);
        if (KERNEL_FLAGS==USE_THREAD_FENCE)
          kernel_solve_thread_fence<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, num_iterations);
        if (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_RED_BLACK))
          kernel_solve_red_black_thread_fence<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, num_iterations);
        if (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_RED_BLACK|USE_SHARED_MEMORY))
          kernel_solve_red_black_thread_fence_shared_memory<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, num_iterations);
        if (KERNEL_FLAGS==(USE_THREAD_FENCE|USE_SHARED_MEMORY))
          kernel_solve_thread_fence_shared_memory<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, num_iterations);

        float error = kernel_solve_get_error(values, expected_values);
        if (OUTPUT_PERFORMANCE) {
          print_tags(tags|SOLVE_TAG);
          printf("[step=%d][gauss_step=%d][error=%f]\n", step, num_iterations, error);
        }
        if (error < EQ_THRESHOLD) break;
      }
      cudaFree(saved_base);
      cudaFree(saved_values);
      return num_iterations;
    } else if (is_host_iterative()) {
      for (int num_iterations = 1; num_iterations <= MAX_CONVERGENCE_ITERATIONS; num_iterations++) {
        for (int red_black = 0; red_black < 2; red_black++)
          kernel_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(red_black, base, values, factor, divisor);

        float error = kernel_solve_get_error(values, expected_values);
        if (OUTPUT_PERFORMANCE) {
          print_tags(tags|SOLVE_TAG);
          printf("[step=%d][gauss_step=%d][error=%f]\n", step, num_iterations, error);
        }
        if (error < EQ_THRESHOLD) return num_iterations;
      }
      return MAX_CONVERGENCE_ITERATIONS;
    } else {
      fprintf(stderr, "Invalid kernel flags: %d\n", KERNEL_FLAGS);
      exit(EXIT_FAILURE);
    }
  } else {
    if (KERNEL_FLAGS == USE_NO_BLOCK_SYNC) {
      kernel_solve_no_block_sync<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, GAUSS_SEIDEL_ITERATIONS);
    } else if (KERNEL_FLAGS == USE_THREAD_FENCE) {
      kernel_solve_thread_fence<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, GAUSS_SEIDEL_ITERATIONS);
    } else if (KERNEL_FLAGS == (USE_THREAD_FENCE|USE_RED_BLACK)) {
      kernel_solve_red_black_thread_fence<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, GAUSS_SEIDEL_ITERATIONS);
    } else if (KERNEL_FLAGS == (USE_THREAD_FENCE|USE_SHARED_MEMORY)) {
      kernel_solve_thread_fence_shared_memory<<<GRID_DIM, BLOCK_DIM>>>(base, values, factor, divisor, GAUSS_SEIDEL_ITERATIONS);
    } else if (KERNEL_FLAGS == USE_RED_BLACK) {
      for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
        for (int red_black = 0; red_black < 2; red_black++)
          kernel_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(red_black, base, values, factor, divisor);
      }
    } else {
      fprintf(stderr, "Invalid kernel flags: %d\n", KERNEL_FLAGS);
      exit(EXIT_FAILURE);
    }
    return GAUSS_SEIDEL_ITERATIONS;
  }
}