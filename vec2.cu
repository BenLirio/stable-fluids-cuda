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
    ((u.x%WIDTH)+WIDTH) % WIDTH,
    ((u.y%HEIGHT)+HEIGHT) % HEIGHT
  );
}