#ifndef STABLE_FLUIDS_CUDA_VEC2_H
#define STABLE_FLUIDS_CUDA_VEC2_H

struct vec2 {
  float x;
  float y;
};
typedef struct vec2 vec2;
#define vec2(x, y) ((vec2) { x, y })

struct idx2 {
  int x;
  int y;
};
typedef struct idx2 idx2;
#define idx2(x, y) ((idx2) { x, y })
idx2 wrap_idx2(idx2 u);
idx2 idx2_add(idx2, idx2);

#endif //STABLE_FLUIDS_CUDA_VEC2_H