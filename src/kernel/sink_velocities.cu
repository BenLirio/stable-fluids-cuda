#include <kernel/sink_velocities.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>

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