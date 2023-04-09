#include "vec2.h"
#include "config.h"

// vec2 wrap_vec2(vec2 u) {
//   return vec2(
//     u.x - WIDTH*floor(u.x/WIDTH),
//     u.y - HEIGHT*floor(u.y/HEIGHT)
//   );
// }

idx2 wrap_idx2(idx2 u) {
  return idx2(
    ((((u.x-1)%WIDTH)+WIDTH) % WIDTH)+1,
    ((((u.y-1)%HEIGHT)+HEIGHT) % HEIGHT)+1
  );
}

idx2 idx2_add(idx2 u, idx2 v) {
  return idx2(u.x+v.x, u.y+v.y);
}