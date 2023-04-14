#ifndef STABLE_FLUIDS_CUDA_IDX2_H
#define STABLE_FLUIDS_CUDA_IDX2_H

#define IDX2(idx2) ((idx2.y-1) * WIDTH + (idx2.x-1))

struct idx2 {
  int x;
  int y;
};
typedef struct idx2 idx2;
#define idx2(x, y) ((idx2) { x, y })
idx2 wrap_idx2(idx2);
idx2 idx2_add(idx2, idx2);

#endif //STABLE_FLUIDS_CUDA_IDX2_H