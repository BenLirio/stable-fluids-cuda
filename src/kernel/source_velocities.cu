#include <kernel/source_velocities.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>

#include <util/type_casting.cuh>
#include <cuda_runtime.h>

__global__ void kernel_source_velocities_single_block(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step) {
  idx2 idx = idx2(
    threadIdx.x + 1,
    threadIdx.y + 1
  );

  vec2 center = vec2((WIDTH/2.0)+0.5, (HEIGHT/2.0)+0.5);
  vec2 position = vec2_of_idx2(idx);
  float distance = vec2_scaled_dist(center, position);
  float magnitude = 1.0/(distance*distance);
  float percent_complete = (float)current_step / (float)NUM_STEPS;
  float x_magnitude = magnitude*cos(percent_complete*M_PI*VELOCITY_SPIN_RATE);
  float y_magnitude = magnitude*sin(percent_complete*M_PI*VELOCITY_SPIN_RATE);

  x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  previous_x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  previous_y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
}

__global__ void kernel_source_velocities_no_optimization(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step) {
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;

  vec2 center = vec2((WIDTH/2.0)+0.5, (HEIGHT/2.0)+0.5);
  vec2 position = vec2_of_idx2(idx);
  float distance = vec2_scaled_dist(center, position);
  float magnitude = 1.0/(distance*distance);
  float x_magnitude = magnitude*cos(current_step*TIME_STEP*M_PI*VELOCITY_SPIN_RATE);
  float y_magnitude = magnitude*sin(current_step*TIME_STEP*M_PI*VELOCITY_SPIN_RATE);
  

  x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  previous_x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
  previous_y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP*VELOCITY_SOURCE_MAGNITUDE;
}

void (*kernel_source_velocities)(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step) = kernel_source_velocities_no_optimization;

void kernel_source_velocities_wrapper(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step) {
  size_t number_of_bytes = sizeof(float)*N;

  float *device_previous_x_velocities;
  float *device_x_velocities;
  float *device_previous_y_velocities;
  float *device_y_velocities;

  cudaMalloc(&device_previous_x_velocities, number_of_bytes);
  cudaMalloc(&device_x_velocities, number_of_bytes);
  cudaMalloc(&device_previous_y_velocities, number_of_bytes);
  cudaMalloc(&device_y_velocities, number_of_bytes);

  cudaMemcpy(device_previous_x_velocities, previous_x_velocities, number_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_x_velocities, x_velocities, number_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_previous_y_velocities, previous_y_velocities, number_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_y_velocities, y_velocities, number_of_bytes, cudaMemcpyHostToDevice);

  kernel_source_velocities<<<1, dim3(WIDTH, HEIGHT)>>>(device_previous_x_velocities, device_previous_y_velocities, device_x_velocities, device_y_velocities, current_step);

  cudaMemcpy(previous_x_velocities, device_previous_x_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(x_velocities, device_x_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_y_velocities, device_previous_y_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(y_velocities, device_y_velocities, number_of_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_previous_x_velocities);
  cudaFree(device_x_velocities);
  cudaFree(device_previous_y_velocities);
  cudaFree(device_y_velocities);
}