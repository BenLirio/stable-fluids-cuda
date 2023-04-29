#include <kernel/sink_colors.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>

__global__ void kernel_sink_colors_no_optimization(float *previous_colors, float *colors) {
  float alpha = (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  if (idx.x > WIDTH || idx.y > HEIGHT) return;
  colors[IDX2(idx)] *= alpha;
  previous_colors[IDX2(idx)] *= alpha;
}

void (*kernel_sink_colors)(float* previous_colors, float* colors) = kernel_sink_colors_no_optimization;
