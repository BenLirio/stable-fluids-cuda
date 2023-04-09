#ifndef STABLE_FLUIDS_CUDA_VEC2_H
#define STABLE_FLUIDS_CUDA_VEC2_H

struct vec2 {
  float x;
  float y;
};
typedef struct vec2 vec2;
#define vec2(x, y) ((vec2) { x, y })
vec2 wrap_vec2(vec2);

struct idx2 {
  int x;
  int y;
};
typedef struct idx2 idx2;
#define idx2(x, y) ((idx2) { x, y })
idx2 wrap_idx2(idx2);
idx2 idx2_add(idx2, idx2);
idx2 wrap_idx2_0_offset(idx2);

#endif //STABLE_FLUIDS_CUDA_VEC2_H