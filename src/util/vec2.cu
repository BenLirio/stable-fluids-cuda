#include <util/vec2.cuh>

#include <util/macros.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

__device__ __host__ float positive_fmod(float x, float m) {
  float r = fmod(x, m);
  return r < 0 ? r + m : r;
}

__device__ __host__ vec2 vec2_wrap(vec2 u) {
  float offset = 0.5;
  return vec2(
    positive_fmod(u.x-offset, (float)WIDTH) + offset,
    positive_fmod(u.y-offset, (float)HEIGHT) + offset
  );
}

__device__ __host__ vec2 vec2_add(vec2 u, vec2 v) {
  return vec2_wrap(vec2(u.x + v.x, u.y + v.y));
}

__device__ __host__ vec2 vec2_scale(float s, vec2 u) {
  return vec2_wrap(vec2(u.x * s, u.y * s));
}

__device__ __host__ float vec2_x_dist(vec2 u, vec2 v) {
  float dx = fabs(u.x - v.x);
  if (dx > WIDTH - dx) dx = WIDTH - dx;
  return dx;
}

__device__ __host__ float vec2_y_dist(vec2 u, vec2 v) {
  float dy = fabs(u.y - v.y);
  if (dy > HEIGHT - dy) dy = HEIGHT - dy;
  return dy;
}

__device__ __host__ float vec2_dist(vec2 u, vec2 v) {
  float dx = vec2_x_dist(u, v);
  float dy = vec2_y_dist(u, v);
  return sqrt(dx*dx + dy*dy);
}


__device__ __host__ float vec2_scaled_dist(vec2 u, vec2 v) {
  float dx = fabs(u.x - v.x);
  if (dx > WIDTH - dx) dx = WIDTH - dx;
  dx = dx/((float) WIDTH);
  float dy = fabs(u.y - v.y);
  if (dy > HEIGHT - dy) dy = HEIGHT - dy;
  dy = dy/((float) HEIGHT);
  return sqrt(dx*dx + dy*dy);
}