#include "vec2.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

vec2 wrap_vec2(vec2 u) {
  float F_WIDTH = (float) WIDTH;
  float F_HEIGHT = (float) HEIGHT;
  float x = u.x;
  while (x < 0.5) { x += F_WIDTH; }
  while (x > (F_WIDTH+0.5)) { x -= F_WIDTH; }
  float y = u.y;
  while (y < 0.5) { y += F_HEIGHT; }
  while (y > (F_HEIGHT+0.5)) { y -= F_HEIGHT; }
  if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (x < 0.5 || x > (F_WIDTH+0.5))) {
    fprintf(stderr, "wrap_vec2: Expected x to be in [0.5, %f], got %f\n", F_WIDTH+0.5, x);
  }
  if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (y < 0.5 || y > (F_HEIGHT+0.5))) {
    fprintf(stderr, "wrap_vec2: Expected y to be in [0.5, %f], got %f\n", F_HEIGHT+0.5, y);
  }
  if (ASSERTIONS_ENABLED) assert(x >= 0.5f && x <= (F_WIDTH+0.5f));
  if (ASSERTIONS_ENABLED) assert(y >= 0.5f && y <= (F_HEIGHT+0.5f));
  return vec2(x, y);
}

float vec2_dist(vec2 u, vec2 v) {
  float dx = fabs(u.x - v.x);
  if (dx > WIDTH - dx) dx = WIDTH - dx;
  dx = dx/((float) WIDTH);
  float dy = fabs(u.y - v.y);
  if (dy > HEIGHT - dy) dy = HEIGHT - dy;
  dy = dy/((float) HEIGHT);
  return sqrt(dx*dx + dy*dy);
}

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