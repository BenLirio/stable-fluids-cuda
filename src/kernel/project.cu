#include <kernel/advect.cuh>
#include <util/macros.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/type_casting.cuh>
#include <util/derivative.cuh>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <kernel/solve.cuh>
#include <gold/solve.cuh>


// __global__ void kernel_project_solve_red_black_naive(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red) {
//   idx2 idx = idx2(
//     blockIdx.x*blockDim.x + threadIdx.x + 1,
//     blockIdx.y*blockDim.y + threadIdx.y + 1
//   );
//   if (idx.x > WIDTH || idx.y > HEIGHT) return;
//   if (idx.x % 2 == (idx.y + red) % 2) return;
//   pressures[IDX2(idx)] = (divergences[IDX2(idx)] + (
//       pressures[IDX2(idx2_add(idx, idx2(0, 1)))] +
//       pressures[IDX2(idx2_add(idx, idx2(0, -1)))] +
//       pressures[IDX2(idx2_add(idx, idx2(1, 0)))] +
//       pressures[IDX2(idx2_add(idx, idx2(-1, 0)))]
//   )) / 4;
// }

// __global__ void kernel_project_solve_red_black_shared(float *x_velocities, float *y_velocities, float *pressures, float *divergences, int red) {
//   __shared__ float shared_pressures[BLOCK_SIZE+2][BLOCK_SIZE+2];
//   idx2 idx = idx2(
//     blockIdx.x*blockDim.x + threadIdx.x + 1,
//     blockIdx.y*blockDim.y + threadIdx.y + 1
//   );
//   int x = threadIdx.x + 1;
//   int y = threadIdx.y + 1;
//   if (idx.x > WIDTH || idx.y > HEIGHT) return;
//   float divergence;

//   if (idx.x % 2 == (idx.y + red) % 2) {
//     shared_pressures[y+0][x+0] = pressures[IDX2(idx)];
//     return;
//   } else {
//     divergence = divergences[IDX2(idx)];
//     if (x == 1)           shared_pressures[y+0][x-1] = pressures[IDX2(idx2_add(idx, idx2(-1, +0)))];
//     if (x == BLOCK_SIZE)  shared_pressures[y+0][x+1] = pressures[IDX2(idx2_add(idx, idx2(+1, +0)))];
//     if (y == 1)           shared_pressures[y-1][x+0] = pressures[IDX2(idx2_add(idx, idx2(+0, -1)))];
//     if (y == BLOCK_SIZE)  shared_pressures[y+1][x+0] = pressures[IDX2(idx2_add(idx, idx2(+0, +1)))];
//   }

//   __syncthreads();
//   pressures[IDX2(idx)] = (divergence + (
//     shared_pressures[y+0][x+1] +
//     shared_pressures[y+0][x-1] +
//     shared_pressures[y+1][x+0] +
//     shared_pressures[y-1][x+0]
//   )) / 4;
// }

__global__ void kernel_project_prepare(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  float x_velocity_derivative = get_x_derivative(x_velocities, idx);
  float y_velocity_derivative = get_y_derivative(y_velocities, idx);
  divergences[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
  pressures[IDX2(idx)] = 0.0f;
}

__global__ void kernel_project_write(float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  float h = 1.0f / sqrt((float)N);
  idx2 idx = idx2(
    blockIdx.x*blockDim.x + threadIdx.x + 1,
    blockIdx.y*blockDim.y + threadIdx.y + 1
  );
  x_velocities[IDX2(idx)] -= get_x_derivative(pressures, idx) / (2*h);
  y_velocities[IDX2(idx)] -= get_y_derivative(pressures, idx) / (2*h);
}

void kernel_project_wrapper(int step, float *x_velocities, float *y_velocities, float *pressures, float *divergences) {
  kernel_project_prepare<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences);

  float factor = 1.0f;
  float divisor = 4.0f;
  if (OUTPUT&OUTPUT_SOLVE_ERROR) {
    float *expected_values;
    cudaMalloc(&expected_values, N*sizeof(float));
    gold_solve_wrapper(expected_values, divergences, pressures, factor, divisor);
    kernel_solve(step, divergences, pressures, expected_values, factor, divisor, PROJECT_TAG);
    cudaFree(expected_values);
  } else {
    kernel_solve(step, divergences, pressures, NULL, factor, divisor, PROJECT_TAG);
  }

  kernel_project_write<<<GRID_DIM, BLOCK_DIM>>>(x_velocities, y_velocities, pressures, divergences);
}