#ifndef STABLE_FLUIDS_CUDA_SINK_COLORS_H_
#define STABLE_FLUIDS_CUDA_SINK_COLORS_H_
#include <cuda_runtime.h>

extern void (*kernel_sink_colors)(float* previous_colors, float* colors);
void kernel_sink_colors_wrapper(float* previous_colors, float* colors);

#endif //STABLE_FLUIDS_CUDA_SINK_COLORS_H_