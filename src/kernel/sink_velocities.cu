#include <kernel/sink_velocities.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>


__global__ void kernel_sink_velocities_single_block(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) {
  float alpha = (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
  idx2 idx = idx2(
    threadIdx.x + 1,
    threadIdx.y + 1
  );
  previous_x_velocities[IDX2(idx)] *= alpha;
  previous_y_velocities[IDX2(idx)] *= alpha;
  x_velocities[IDX2(idx)] *= alpha;
  y_velocities[IDX2(idx)] *= alpha;
}

__global__ void kernel_sink_velocities_no_optimization(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) {
  float alpha = (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;
  previous_x_velocities[IDX2(idx)] *= alpha;
  previous_y_velocities[IDX2(idx)] *= alpha;
  x_velocities[IDX2(idx)] *= alpha;
  y_velocities[IDX2(idx)] *= alpha;
}

void (*kernel_sink_velocities)(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) = kernel_sink_velocities_no_optimization;

void kernel_sink_velocities_wrapper(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) {
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

  kernel_sink_velocities<<<1, dim3(WIDTH, HEIGHT)>>>(device_previous_x_velocities, device_previous_y_velocities, device_x_velocities, device_y_velocities);

  cudaMemcpy(previous_x_velocities, device_previous_x_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(x_velocities, device_x_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_y_velocities, device_previous_y_velocities, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(y_velocities, device_y_velocities, number_of_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_previous_x_velocities);
  cudaFree(device_x_velocities);
  cudaFree(device_previous_y_velocities);
  cudaFree(device_y_velocities);
}