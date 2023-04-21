#include <kernel/sink_colors.cuh>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/idx2.cuh>

__global__ void kernel_sink_colors(float *previous_colors, float *colors) {
  float alpha = (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
  idx2 idx = idx2(
    threadIdx.x + 1,
    threadIdx.y + 1
  );
  colors[IDX2(idx)] *= alpha;
  previous_colors[IDX2(idx)] *= alpha;
}

void kernel_sink_colors_wrapper(float *previous_colors, float *colors) {
  float *device_previous_colors;
  float *device_colors;
  size_t number_of_bytes = sizeof(float)*N;

  cudaMalloc(&device_previous_colors, number_of_bytes);
  cudaMalloc(&device_colors, number_of_bytes);

  cudaMemcpy(device_previous_colors, previous_colors, number_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_colors, colors, number_of_bytes, cudaMemcpyHostToDevice);

  kernel_sink_colors<<<1, dim3(WIDTH, HEIGHT)>>>(device_previous_colors, device_colors);

  cudaMemcpy(previous_colors, device_previous_colors, number_of_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(colors, device_colors, number_of_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_previous_colors);
  cudaFree(device_colors);
}