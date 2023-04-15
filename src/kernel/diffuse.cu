#include <util/compile_options.h>
#include <util/macros.h>
#include <stdio.h>
#include <util/idx2.cuh>


// void gold_diffuse(float *previous_values, float *values, float rate) {
//   float factor = TIME_STEP*rate*WIDTH*HEIGHT;
//   for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
//     for (int y = 1; y <= HEIGHT; y++) {
//       for (int x = 1; x <= WIDTH; x++) {
//         idx2 idx = idx2(x, y);
//         float sum = 0;
//         for (int i = 0; i < NUM_NEIGHBORS; i++) {
//           idx2 neighbor_idx = idx2_wrap(idx2(
//               idx.x + adjancent_offsets[i].x,
//               idx.y + adjancent_offsets[i].y
//             ));
//           sum += values[IDX2(neighbor_idx)];
//         }
//         values[IDX2(idx)] = (previous_values[IDX2(idx)] + factor*sum) / (1 + 4*factor);
//       }
//     }
//   }
// }

__global__ void kernel_diffuse(float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  idx2 idx = idx2(x, y);
  idx2_add(idx, idx2(1, 0));
  idx2_add(idx, idx);
  foo();
  idx2 right_idx = idx2_add(idx, idx2(1, 0));
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    float next_value = (
      previous_values[IDX2(idx)] +
      factor*(
        values[IDX2(idx2_add(idx, idx2(1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
        values[IDX2(idx2_add(idx, idx2(0, 1)))] +
        values[IDX2(idx2_add(idx, idx2(0, -1)))]
      )
    ) / (1 + 4*factor);
    __syncthreads();
    values[IDX2(idx)] = next_value;
    __syncthreads();
  }
}

void kernel_diffuse_wrapper(float *previous_values, float *values, float rate) {
  float *d_previous_values, *d_values;
  cudaMalloc(&d_previous_values, sizeof(float)*N);
  cudaMalloc(&d_values, sizeof(float)*N);

  cudaMemcpy(d_previous_values, previous_values, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, sizeof(float)*N, cudaMemcpyHostToDevice);
  kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(d_previous_values, d_values, rate);

  cudaMemcpy(values, d_values, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(previous_values, d_previous_values, sizeof(float)*N, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_previous_values);
}