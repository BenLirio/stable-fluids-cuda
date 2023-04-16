#include <util/derivative.cuh>
#include <util/idx2.cuh>

__device__ __host__ float get_x_derivative(float *x_velocities, idx2 idx) {
  idx2 next_idx = idx2_add(idx, idx2(1, 0));
  idx2 previous_idx = idx2_add(idx, idx2(-1, 0));
  return x_velocities[IDX2(next_idx)] - x_velocities[IDX2(previous_idx)];
}
__device__ __host__ float get_y_derivative(float *y_velocities, idx2 idx) {
  idx2 next_idx = idx2_add(idx, idx2(0, 1));
  idx2 previous_idx = idx2_add(idx, idx2(0, -1));
  return y_velocities[IDX2(next_idx)] - y_velocities[IDX2(previous_idx)];
}
