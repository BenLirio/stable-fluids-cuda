#include <kernel/source_velocities.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>

#include <util/type_casting.cuh>
#include <cuda_runtime.h>

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