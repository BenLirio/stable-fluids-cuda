#include <util/compile_options.h>
#include <util/macros.h>
#include <stdio.h>
#include <util/idx2.cuh>
#include <cuda_runtime.h>

__global__ void kernel_diffuse_single_block(float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  int x = threadIdx.x+1;
  int y = threadIdx.y+1;
  idx2 idx = idx2(x, y);
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    float next_value = (
      previous_values[IDX2(idx)] +
      factor*(
        values[IDX2(idx2_add(idx, idx2(1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(0, 1)))] +
        values[IDX2(idx2_add(idx, idx2(0, -1)))]
      )
    ) / (1 + 4*factor);
    __syncthreads();
    values[IDX2(idx)] = next_value;
    __syncthreads();
  }
}

__global__ void kernel_diffuse_no_optimization(float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    float next_value = (
      previous_values[IDX2(idx)] +
      factor*(
        values[IDX2(idx2_add(idx, idx2(1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(0, 1)))] +
        values[IDX2(idx2_add(idx, idx2(0, -1)))]
      )
    ) / (1 + 4*factor);
    __syncthreads();
    if (idx.x >= 1 && idx.x <= WIDTH && idx.y >= 1 && idx.y <= HEIGHT)
      values[IDX2(idx)] = next_value;
    __syncthreads();
  }
}

void (*kernel_diffuse)(float *previous_values, float *values, float rate) = kernel_diffuse_no_optimization;

__global__ void kernel_diffuse_red_black_naive(float *previous_values, float *values, float rate, int red) {
  float factor = TIME_STEP*rate*N;
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;
  if (idx.x % 2 == (idx.y+red) % 2) return;
  values[IDX2(idx)] = (
    previous_values[IDX2(idx)] +
    factor*(
      values[IDX2(idx2_add(idx, idx2(1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(0, 1)))] +
      values[IDX2(idx2_add(idx, idx2(0, -1)))]
    )
  ) / (1 + 4*factor);
}

__global__ void kernel_diffuse_red_black_shared(float *previous_values, float *values, float rate, int red) {
  float factor = TIME_STEP*rate*N;
  __shared__ float shared_values[BLOCK_SIZE+2][BLOCK_SIZE+2];

  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;

  int x = threadIdx.x+1;
  int y = threadIdx.y+1;

                        shared_values[x+0][y+0] = values[IDX2(idx)];
  if (x == 1)           shared_values[x-1][y+0] = values[IDX2(idx2_add(idx, idx2(-1, +0)))];
  if (x == BLOCK_SIZE)  shared_values[x+1][y+0] = values[IDX2(idx2_add(idx, idx2(+1, +0)))];
  if (y == 1)           shared_values[x+0][y-1] = values[IDX2(idx2_add(idx, idx2(+0, -1)))];
  if (y == BLOCK_SIZE)  shared_values[x+0][y+1] = values[IDX2(idx2_add(idx, idx2(+0, +1)))];

  if (idx.x % 2 == (idx.y+red) % 2) return;
  __syncthreads();

  values[IDX2(idx)] = (
    previous_values[IDX2(idx)] +
    factor*(
      shared_values[x+1][y+0] +
      shared_values[x-1][y+0] +
      shared_values[x+0][y+1] +
      shared_values[x+0][y-1]
    )
  ) / (1 + 4*factor);
}

__global__ void kernel_diffuse_red_black_shared_and_neighbor_map(float *previous_values, float *values, float rate, int red) {
  float factor = TIME_STEP*rate*N;
  __shared__ float shared_values[BLOCK_SIZE+2][BLOCK_SIZE+2];

  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;

  int x = threadIdx.x+1;
  int y = threadIdx.y+1;

                        shared_values[x+0][y+0] = values[IDX2(idx)];
  if (x == 1)           shared_values[x-1][y+0] = values[IDX2(idx2_add(idx, idx2(-1, +0)))];
  if (x == BLOCK_SIZE)  shared_values[x+1][y+0] = values[IDX2(idx2_add(idx, idx2(+1, +0)))];
  if (y == 1)           shared_values[x+0][y-1] = values[IDX2(idx2_add(idx, idx2(+0, -1)))];
  if (y == BLOCK_SIZE)  shared_values[x+0][y+1] = values[IDX2(idx2_add(idx, idx2(+0, +1)))];

  if (idx.x % 2 == (idx.y+red) % 2) return;
  __syncthreads();

  values[IDX2(idx)] = (
    previous_values[IDX2(idx)] +
    factor*(
      shared_values[x+1][y+0] +
      shared_values[x-1][y+0] +
      shared_values[x+0][y+1] +
      shared_values[x+0][y-1]
    )
  ) / (1 + 4*factor);
}

void (*kernel_diffuse_red_black)(float *previous_values, float *values, float rate, int red) = kernel_diffuse_red_black_shared;

void kernel_diffuse_wrapper(float *previous_values, float *values, float rate) {
  for (int i = 0; i < GAUSS_SEIDEL_ITERATIONS; i++) {
    kernel_diffuse_red_black<<<GRID_DIM, BLOCK_DIM>>>(previous_values, values, rate, RED);
    kernel_diffuse_red_black<<<GRID_DIM, BLOCK_DIM>>>(previous_values, values, rate, BLACK);
  }
}

void kernel_diffuse_test_harness(float *previous_values, float *values, float rate) {
  float *d_previous_values, *d_values;
  cudaMalloc(&d_previous_values, sizeof(float)*N);
  cudaMalloc(&d_values, sizeof(float)*N);

  cudaMemcpy(d_previous_values, previous_values, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(float)*N, cudaMemcpyHostToDevice);

  kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(d_previous_values, d_values, rate);

  cudaMemcpy(values, d_values, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_values, d_previous_values, sizeof(float)*N, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_previous_values);
}