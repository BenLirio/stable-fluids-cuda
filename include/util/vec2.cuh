#ifndef STABLE_FLUIDS_CUDA_VEC2_H
#define STABLE_FLUIDS_CUDA_VEC2_H

#include <cuda_runtime.h>

struct vec2 {
  float x;
  float y;
};
typedef struct vec2 vec2;
#define vec2(x, y) ((vec2) { x, y })
__host__ __device__ vec2 vec2_wrap(vec2);
__host__ __device__ vec2 vec2_add(vec2, vec2);
__host__ __device__ float vec2_x_dist(vec2, vec2);
__host__ __device__ float vec2_y_dist(vec2, vec2);
__host__ __device__ float vec2_dist(vec2, vec2);
__host__ __device__ vec2 vec2_scale(float, vec2);
__host__ __device__ float vec2_scaled_dist(vec2, vec2);

#endif //STABLE_FLUIDS_CUDA_VEC2_H