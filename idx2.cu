#include "idx2.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

idx2 wrap_idx2(idx2 u) {
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