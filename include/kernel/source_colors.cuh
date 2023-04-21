#ifndef STABLE_FLUIDS_CUDA_SOURCE_COLORS_H_
#define STABLE_FLUIDS_CUDA_SOURCE_COLORS_H_

#include <cuda_runtime.h>

__global__ void kernel_source_colors(float *previous_colors, float *colors);
void kernel_source_colors_wrapper(float *previous_colors, float *colors);

#endif //STABLE_FLUIDS_CUDA_SOURCE_COLORS_H_