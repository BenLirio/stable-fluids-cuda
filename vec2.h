#ifndef STABLE_FLUIDS_CUDA_VEC2_H
#define STABLE_FLUIDS_CUDA_VEC2_H

struct vec2 {
  float x;
  float y;
};
typedef struct vec2 vec2;
#define vec2(x, y) ((vec2) { x, y })
vec2 wrap_vec2(vec2);
float vec2_dist(vec2, vec2);

#endif //STABLE_FLUIDS_CUDA_VEC2_H