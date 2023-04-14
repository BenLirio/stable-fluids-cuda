#include "idx2.h"
#include "compile_options.h"
#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

idx2 idx2_wrap(idx2 u) {
  int x = u.x;
  while (x < 1) { x += WIDTH; }
  while (x > WIDTH) { x -= WIDTH; }
  int y = u.y;
  while (y < 1) { y += HEIGHT; }
  while (y > HEIGHT) { y -= HEIGHT; }
  return idx2(x, y);
}

idx2 idx2_add(idx2 u, idx2 v) {
  return idx2(u.x+v.x, u.y+v.y);
}

idx2 _adjancent_offsets[NUM_NEIGHBORS] = {
    idx2(0, 1),
    idx2(1, 0),
    idx2(0, -1),
    idx2(-1, 0),
};
idx2 *adjancent_offsets = _adjancent_offsets;

idx2 _lower_right_square_offsets[NUM_NEIGHBORS] = {
  idx2(0, 0),
  idx2(0, 1),
  idx2(1, 0),
  idx2(1, 1),
};

idx2 *lower_right_square_offsets = _lower_right_square_offsets;