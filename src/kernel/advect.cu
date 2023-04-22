#include <kernel/advect.cuh>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/type_casting.cuh>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = TIME_STEP*sqrt((float)N);
  int x = threadIdx.x + 1;
  int y = threadIdx.y + 1;
  idx2 idx = idx2(x, y);
  vec2 pos = vec2_of_idx2(idx);
  vec2 velocity = vec2(x_velocities[IDX2(idx)], y_velocities[IDX2(idx)]);
  vec2 scaled_velocity = vec2_scale(-alpha, velocity);
  vec2 pos_offset_by_velocity = vec2_add(pos, scaled_velocity);
  idx2 idx_offset_by_velocity = idx2_of_vec2(pos_offset_by_velocity);
  vec2 pos_offset_by_velocity_floored = vec2_of_idx2(idx_offset_by_velocity);
  float wx0 = vec2_x_dist(pos_offset_by_velocity, pos_offset_by_velocity_floored);
  float wx1 = 1 - wx0;
  float wy0 = vec2_y_dist(pos_offset_by_velocity, pos_offset_by_velocity_floored);
  float wy1 = 1 - wy0;
  values[IDX2(idx)] = (
      wx1*wy1*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(0, 0)))]
    + wx1*wy0*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(0, 1)))]
    + wx0*wy1*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(1, 0)))]
    + wx0*wy0*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(1, 1)))]
  );
}

void kernel_advect_wrapper(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float *d_previous_values, *d_values, *d_x_velocities, *d_y_velocities;
  cudaMalloc(&d_previous_values, sizeof(float)*N);
  cudaMalloc(&d_values, sizeof(float)*N);
  cudaMalloc(&d_x_velocities, sizeof(float)*N);
  cudaMalloc(&d_y_velocities, sizeof(float)*N);

  cudaMemcpy(d_previous_values, previous_values, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_velocities, x_velocities, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_velocities, y_velocities, sizeof(float)*N, cudaMemcpyHostToDevice);

  kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(d_previous_values, d_values, d_x_velocities, d_y_velocities);

  cudaMemcpy(values, d_values, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_values, d_previous_values, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(x_velocities, d_x_velocities, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(y_velocities, d_y_velocities, sizeof(float)*N, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_previous_values);
  cudaFree(d_x_velocities);
  cudaFree(d_y_velocities);
}