#include "type_casting.h"
#include <vec2.h>
#include <idx2.h>

idx2 idx2_of_vec2(vec2 v) {
  return idx2_wrap(idx2((int)v.x, (int)v.y));
}

vec2 vec2_of_idx2(idx2 idx) {
  return vec2_wrap(vec2((float)idx.x, (float)idx.y));
}