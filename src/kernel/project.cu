#include <kernel/advect.cuh>
#include <util/macros.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/type_casting.cuh>
#include <util/derivative.cuh>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_project_single_block(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  int x = threadIdx.x + 1;
  int y = threadIdx.y + 1;
  idx2 idx = idx2(x, y);

  float x_velocity_derivative = get_x_derivative(x_velocities, idx);
  float y_velocity_derivative = get_y_derivative(y_velocities, idx);
  divergences[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
  pressures[IDX2(idx)] = 0.0f;

  __syncthreads();
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    float sum = (
      pressures[IDX2(idx2_add(idx, idx2(0, 1)))] +
      pressures[IDX2(idx2_add(idx, idx2(0, -1)))] +
      pressures[IDX2(idx2_add(idx, idx2(1, 0)))] +
      pressures[IDX2(idx2_add(idx, idx2(-1, 0)))]
    );
    __syncthreads();
    pressures[IDX2(idx)] = (divergences[IDX2(idx)] + sum) / 4;
    __syncthreads();
  }

  x_velocities[IDX2(idx)] -= get_x_derivative(pressures, idx) / (2*h);
  y_velocities[IDX2(idx)] -= get_y_derivative(pressures, idx) / (2*h);
}

__global__ void kernel_project_no_optimization(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );

  float x_velocity_derivative = get_x_derivative(x_velocities, idx);
  float y_velocity_derivative = get_y_derivative(y_velocities, idx);
  divergences[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
  pressures[IDX2(idx)] = 0.0f;

  __syncthreads();
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    float sum = (
      pressures[IDX2(idx2_add(idx, idx2(0, 1)))] +
      pressures[IDX2(idx2_add(idx, idx2(0, -1)))] +
      pressures[IDX2(idx2_add(idx, idx2(1, 0)))] +
      pressures[IDX2(idx2_add(idx, idx2(-1, 0)))]
    );
    __syncthreads();
    if (idx.x <= WIDTH && idx.y <= HEIGHT)
      pressures[IDX2(idx)] = (divergences[IDX2(idx)] + sum) / 4;
    __syncthreads();
  }

  x_velocities[IDX2(idx)] -= get_x_derivative(pressures, idx) / (2*h);
  y_velocities[IDX2(idx)] -= get_y_derivative(pressures, idx) / (2*h);
}

void (*kernel_project)(float *x_velocities, float *y_velocities, float *pressure, float *divergence) = kernel_project_no_optimization;

__global__ void kernel_project_prepare(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  float x_velocity_derivative = get_x_derivative(x_velocities, idx);
  float y_velocity_derivative = get_y_derivative(y_velocities, idx);
  divergences[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
  pressures[IDX2(idx)] = 0.0f;
}

__global__ void kernel_project_solve_red_black_naive(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red) {
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;
  if (idx.x % 2 == (idx.y + red) % 2) return;
  pressures[IDX2(idx)] = (divergences[IDX2(idx)] + (
      pressures[IDX2(idx2_add(idx, idx2(0, 1)))] +
      pressures[IDX2(idx2_add(idx, idx2(0, -1)))] +
      pressures[IDX2(idx2_add(idx, idx2(1, 0)))] +
      pressures[IDX2(idx2_add(idx, idx2(-1, 0)))]
  )) / 4;
}

__global__ void kernel_project_solve_red_black_shared(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red) {
  __shared__ float shared_pressures[BLOCK_SIZE+2][BLOCK_SIZE+2];
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  int x = threadIdx.x + 1;
  int y = threadIdx.y + 1;
  if (idx.x > WIDTH || idx.y > HEIGHT) return;
  float divergence;

  if (idx.x % 2 == (idx.y + red) % 2) {
    shared_pressures[y+0][x+0] = pressures[IDX2(idx)];
    return;
  } else {
    divergence = divergences[IDX2(idx)];
    if (x == 1)           shared_pressures[y+0][x-1] = pressures[IDX2(idx2_add(idx, idx2(-1, +0)))];
    if (x == BLOCK_SIZE)  shared_pressures[y+0][x+1] = pressures[IDX2(idx2_add(idx, idx2(+1, +0)))];
    if (y == 1)           shared_pressures[y-1][x+0] = pressures[IDX2(idx2_add(idx, idx2(+0, -1)))];
    if (y == BLOCK_SIZE)  shared_pressures[y+1][x+0] = pressures[IDX2(idx2_add(idx, idx2(+0, +1)))];
  }

  __syncthreads();
  pressures[IDX2(idx)] = (divergence + (
    shared_pressures[y+0][x+1] +
    shared_pressures[y+0][x-1] +
    shared_pressures[y+1][x+0] +
    shared_pressures[y-1][x+0]
  )) / 4;
}


__global__ void kernel_project_write(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  x_velocities[IDX2(idx)] -= get_x_derivative(pressures, idx) / (2*h);
  y_velocities[IDX2(idx)] -= get_y_derivative(pressures, idx) / (2*h);
}

void kernel_project_wrapper(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {

  void (*kernel_project_solve_red_black)(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red) = kernel_project_solve_red_black_naive;

  if (KERNEL_FLAGS&USE_SHARED_MEMORY) {
    kernel_project_solve_red_black = kernel_project_solve_red_black_shared;
  }

  kernel_project_prepare<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences);
  for (int i = 0; i < GAUSS_SEIDEL_ITERATIONS; i++) {
    kernel_project_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences, RED);
    kernel_project_solve_red_black<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences, BLACK);
  }
  kernel_project_write<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences);
}

void kernel_project_test_harness(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float *device_x_velocities, *device_y_velocities, *device_pressure, *device_divergences;

  cudaMalloc(&device_x_velocities, sizeof(float)*N);
  cudaMalloc(&device_y_velocities, sizeof(float)*N);
  cudaMalloc(&device_pressure, sizeof(float)*N);
  cudaMalloc(&device_divergences, sizeof(float)*N);

  cudaMemcpy(device_x_velocities, x_velocities, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_y_velocities, y_velocities, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_pressure, pressures, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_divergences, divergences, sizeof(float)*N, cudaMemcpyHostToDevice);

  kernel_project<<<1, dim3(WIDTH, HEIGHT)>>>(device_x_velocities, device_y_velocities, device_pressure, device_divergences);

  cudaMemcpy(x_velocities, device_x_velocities, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(y_velocities, device_y_velocities, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(pressures, device_pressure, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(divergences, device_divergences, sizeof(float)*N, cudaMemcpyDeviceToHost);

  cudaFree(device_x_velocities);
  cudaFree(device_y_velocities);
  cudaFree(device_pressure);
  cudaFree(device_divergences);
}
