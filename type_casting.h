#ifndef STABLE_FLUIDS_CUDA_TYPE_CASTING_H_
#define STABLE_FLUIDS_CUDA_TYPE_CASTING_H_
#include "vec2.h"
#include "idx2.h"

idx2 idx2_of_vec2(vec2);
vec2 vec2_of_idx2(idx2);

#endif