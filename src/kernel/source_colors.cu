#include <kernel/source_colors.cuh>
#include <cuda_runtime.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/type_casting.cuh>
#include <util/macros.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void kernel_source_colors_single_block(float *previous_colors, float *colors) {
  idx2 idx = idx2(
    threadIdx.x + 1,
    threadIdx.y + 1
  );
  vec2 center = vec2((WIDTH/2.0) + 0.5, (HEIGHT/2.0) + 0.5);
  vec2 position = vec2_of_idx2(idx);
  float distance = vec2_scaled_dist(center, position);
  float magnitude = 1.0/(distance*distance);
  colors[IDX2(idx)] += magnitude*TIME_STEP;
  previous_colors[IDX2(idx)] += magnitude*TIME_STEP;
}

__global__ void kernel_source_colors_no_optimization(float *previous_colors, float *colors) {
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT)
    return;
  vec2 center = vec2((WIDTH/2.0) + 0.5, (HEIGHT/2.0) + 0.5);
  vec2 position = vec2_of_idx2(idx);
  float distance = vec2_scaled_dist(center, position);
  float magnitude = 1.0/(distance*distance);
  colors[IDX2(idx)] += magnitude*TIME_STEP;
  previous_colors[IDX2(idx)] += magnitude*TIME_STEP;
}

void (*kernel_source_colors)(float *previous_colors, float *colors) = kernel_source_colors_no_optimization;

void kernel_source_colors_wrapper(float *previous_colors, float *colors) {
  size_t number_of_bytes = N*sizeof(float);

  float *device_previous_colors;
  float *device_colors;

  cudaMalloc(&device_previous_colors, number_of_bytes);
  cudaMalloc(&device_colors, number_of_bytes);

  cudaMemcpy(device_previous_colors, previous_colors, number_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_colors, colors, number_of_bytes, cudaMemcpyHostToDevice);

  kernel_source_colors<<<1, dim3(WIDTH, HEIGHT)>>>(device_previous_colors, device_colors);

  cudaMemcpy(previous_colors, device_previous_colors, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(colors, device_colors, number_of_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_previous_colors);
  cudaFree(device_colors);
}