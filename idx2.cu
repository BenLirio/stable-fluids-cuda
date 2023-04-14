#include "idx2.h"
#include "compile_options.h"
#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


int mod(int n, int m) {
  int r = n % m;
  return r < 0 ? r + m : r;
}

idx2 idx2_wrap(idx2 u) {
  return idx2(
    mod(u.x-1, WIDTH) + 1,
    mod(u.y-1, HEIGHT) + 1
  );
}

idx2 idx2_add(idx2 u, idx2 v) {
  return idx2_wrap(idx2(u.x+v.x, u.y+v.y));
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