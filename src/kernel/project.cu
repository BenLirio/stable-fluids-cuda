#include <kernel/advect.cuh>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/type_casting.cuh>
#include <util/derivative.cuh>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__constant__ idx2 constant_adjacent_offsets[NUM_NEIGHBORS];
__global__ void kernel_project(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
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

void kernel_project_wrapper(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  cudaMemcpyToSymbol(constant_adjacent_offsets, adjancent_offsets, sizeof(idx2)*NUM_NEIGHBORS);
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
