#include <util/macros.h>
#include <stdio.h>
#include <util/idx2.cuh>
#include <cuda_runtime.h>
#include <kernel/solve.cuh>
#include <gold/solve.cuh>

// __global__ void kernel_diffuse_red_black_naive(float *previous_values, float *values, float rate, int red) {
//   float factor = TIME_STEP*rate*N;
//   idx2 idx = idx2(
//     blockIdx.x*blockDim.x + threadIdx.x + 1,
//     blockIdx.y*blockDim.y + threadIdx.y + 1
//   );
//   if (idx.x > WIDTH || idx.y > HEIGHT) return;
//   if (idx.x % 2 == (idx.y+red) % 2) return;
//   values[IDX2(idx)] = (
//     previous_values[IDX2(idx)] +
//     factor*(
//       values[IDX2(idx2_add(idx, idx2(1, 0)))] +
//       values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
//       values[IDX2(idx2_add(idx, idx2(0, 1)))] +
//       values[IDX2(idx2_add(idx, idx2(0, -1)))]
//     )
//   ) / (1 + 4*factor);
// }


// __global__ void kernel_diffuse_red_black_shared(float *previous_values, float *values, float rate, int red) {
//   float factor = TIME_STEP*rate*N;
//   __shared__ float shared_values[BLOCK_SIZE+2][BLOCK_SIZE+2];

//   idx2 idx = idx2(
//     blockIdx.x*blockDim.x + threadIdx.x + 1,
//     blockIdx.y*blockDim.y + threadIdx.y + 1
//   );
//   if (idx.x > WIDTH || idx.y > HEIGHT) return;

//   int x = threadIdx.x+1;
//   int y = threadIdx.y+1;


//   float previous_value;
//   if (idx.x % 2 == (idx.y+red) % 2) {
//     shared_values[x+0][y+0] = values[IDX2(idx)];
//     return;
//   } else {
//     previous_value = previous_values[IDX2(idx)];
//     if (x == 1)           shared_values[x-1][y+0] = values[IDX2(idx2_add(idx, idx2(-1, +0)))];
//     if (x == BLOCK_SIZE)  shared_values[x+1][y+0] = values[IDX2(idx2_add(idx, idx2(+1, +0)))];
//     if (y == 1)           shared_values[x+0][y-1] = values[IDX2(idx2_add(idx, idx2(+0, -1)))];
//     if (y == BLOCK_SIZE)  shared_values[x+0][y+1] = values[IDX2(idx2_add(idx, idx2(+0, +1)))];
//   }
//   __syncthreads();

//   values[IDX2(idx)] = (
//     previous_value +
//     factor*(
//       shared_values[x+1][y+0] +
//       shared_values[x-1][y+0] +
//       shared_values[x+0][y+1] +
//       shared_values[x+0][y-1]
//     )
//   ) / (1 + 4*factor);
// }

// __global__ void kernel_diffuse_red_black_thread_coarsening(float *previous_values, float *values, float rate, int red) {

//   float factor = TIME_STEP*rate*N;
//   int coarsening = 2;

//   idx2 base_idx = idx2(
//     blockIdx.x*(blockDim.x*coarsening) + (threadIdx.x*coarsening) + 1,
//     blockIdx.y*(blockDim.y*coarsening) + (threadIdx.y*coarsening) + 1
//   );

//   for (int y = 0; y < coarsening; y++) {
//     for (int x = 0; x < coarsening; x++) {
//       idx2 idx = idx2_add(base_idx, idx2(x, y));
//       if (idx.x > WIDTH || idx.y > HEIGHT) continue;
//       if (idx.x % 2 == (idx.y+red) % 2) continue;
//       values[IDX2(idx)] = (
//         previous_values[IDX2(idx)] +
//         factor*(
//           values[IDX2(idx2_add(idx, idx2(1, 0)))] +
//           values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
//           values[IDX2(idx2_add(idx, idx2(0, 1)))] +
//           values[IDX2(idx2_add(idx, idx2(0, -1)))]
//         )
//       ) / (1 + 4*factor);
//     }
//   }
// }

void kernel_diffuse_wrapper(int step, float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  float divisor = 1 + 4*factor;
  if (OUTPUT&OUTPUT_SOLVE_ERROR) {
    float *expected_values;
    cudaMalloc(&expected_values, N*sizeof(float));
    gold_solve_wrapper(expected_values, previous_values, values, factor, divisor);
    kernel_solve(step, previous_values, values, expected_values, factor, divisor, DIFFUSE_TAG);
    cudaFree(expected_values);
  } else {
    kernel_solve(step, previous_values, values, NULL, factor, divisor, DIFFUSE_TAG);
  }
}