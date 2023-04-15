#ifndef STABLE_FLUIDS_CUDA_IDX2_H
#define STABLE_FLUIDS_CUDA_IDX2_H

#include <cuda_runtime.h>
#include <util/vec2.h>

#define IDX2(idx2) ((idx2.y-1) * WIDTH + (idx2.x-1))


struct idx2 {
  int x;
  int y;
};
typedef struct idx2 idx2;
#define idx2(x, y) ((idx2) { x, y })
__device__ __host__ idx2 idx2_wrap(idx2);
__device__ __host__ idx2 idx2_add(idx2, idx2);
__device__ int foo();

extern idx2 *adjancent_offsets;
extern idx2 *lower_right_square_offsets;

#endif //STABLE_FLUIDS_CUDA_IDX2_H