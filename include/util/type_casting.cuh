#ifndef STABLE_FLUIDS_CUDA_TYPE_CASTING_H_
#define STABLE_FLUIDS_CUDA_TYPE_CASTING_H_
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <cuda_runtime.h>

__device__ __host__ idx2 idx2_of_vec2(vec2);
__device__ __host__ vec2 vec2_of_idx2(idx2);

#endif