#include <cuda_runtime.h>
#include <util/compile_options.h>
#include <kernel/diffuse.cuh>
#include <kernel/advect.cuh>
#include <kernel/project.cuh>
#include <util/macros.h>
#include <kernel/index.cuh>

void kernel_step(
    float *colors,
    float *previous_colors,
    float *x_velocities,
    float *previous_x_velocities,
    float *y_velocities,
    float *previous_y_velocities,
    float *preasures,
    float *divergences
  ) {
  // density
  // source_colors();
  // sink_colors();
  SWAP(previous_colors, colors);
  kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(previous_colors, colors, DIFFUSION_RATE);
  SWAP(previous_colors, colors);
  kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(previous_colors, colors, x_velocities, y_velocities);

  // velocity
  // source_velocity();
  // sink_velocity();
  SWAP(previous_x_velocities, x_velocities);
  kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(previous_x_velocities, x_velocities, VISCOSITY);
  SWAP(previous_y_velocities, y_velocities);
  kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(previous_y_velocities, y_velocities, VISCOSITY);
  kernel_project<<<1, dim3(WIDTH, HEIGHT)>>>(previous_x_velocities, previous_y_velocities, preasures, divergences);

  SWAP(previous_x_velocities, x_velocities);
  SWAP(previous_y_velocities, y_velocities);
  kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(previous_x_velocities, x_velocities, previous_x_velocities, previous_y_velocities);
  kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(previous_y_velocities, y_velocities, previous_x_velocities, previous_y_velocities);
  kernel_project<<<1, dim3(WIDTH, HEIGHT)>>>(previous_x_velocities, previous_y_velocities, preasures, divergences);
}

void kernel_step_wrapper(
  float *colors,
  float *previous_colors,
  float *x_velocities,
  float *previous_x_velocities,
  float *y_velocities,
  float *previous_y_velocities,
  float *preasures,
  float *divergences
) {
  float *device_colors;
  float *device_previous_colors;
  float *device_x_velocities;
  float *device_previous_x_velocities;
  float *device_y_velocities;
  float *device_previous_y_velocities;
  float *device_preasures;
  float *device_divergences;

  cudaMalloc(&device_colors, N*sizeof(float));
  cudaMalloc(&device_previous_colors, N*sizeof(float));
  cudaMalloc(&device_x_velocities, N*sizeof(float));
  cudaMalloc(&device_previous_x_velocities, N*sizeof(float));
  cudaMalloc(&device_y_velocities, N*sizeof(float));
  cudaMalloc(&device_previous_y_velocities, N*sizeof(float));
  cudaMalloc(&device_preasures, N*sizeof(float));
  cudaMalloc(&device_divergences, N*sizeof(float));

  cudaMemcpy(device_colors, colors, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_previous_colors, previous_colors, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_x_velocities, x_velocities, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_previous_x_velocities, previous_x_velocities, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y_velocities, y_velocities, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_previous_y_velocities, previous_y_velocities, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_preasures, preasures, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_divergences, divergences, N*sizeof(float), cudaMemcpyHostToDevice);

  kernel_step(
    device_colors,
    device_previous_colors,
    device_x_velocities,
    device_previous_x_velocities,
    device_y_velocities,
    device_previous_y_velocities,
    device_preasures,
    device_divergences
  );

  cudaMemcpy(colors, device_colors, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_colors, device_previous_colors, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(x_velocities, device_x_velocities, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_x_velocities, device_previous_x_velocities, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y_velocities, device_y_velocities, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_y_velocities, device_previous_y_velocities, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(preasures, device_preasures, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(divergences, device_divergences, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_colors);
  cudaFree(device_previous_colors);
  cudaFree(device_x_velocities);
  cudaFree(device_previous_x_velocities);
  cudaFree(device_y_velocities);
  cudaFree(device_previous_y_velocities);
  cudaFree(device_preasures);
  cudaFree(device_divergences);
}