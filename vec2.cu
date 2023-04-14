#include "vec2.h"
#include "compile_options.h"
#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

vec2 vec2_wrap(vec2 u) {
  float F_WIDTH = (float) WIDTH;
  float F_HEIGHT = (float) HEIGHT;
  float x = u.x;
  while (x < 0.5) { x += F_WIDTH; }
  while (x > (F_WIDTH+0.5)) { x -= F_WIDTH; }
  float y = u.y;
  while (y < 0.5) { y += F_HEIGHT; }
  while (y > (F_HEIGHT+0.5)) { y -= F_HEIGHT; }
  if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (x < 0.5 || x > (F_WIDTH+0.5))) {
    fprintf(stderr, "vec2_wrap: Expected x to be in [0.5, %f], got %f\n", F_WIDTH+0.5, x);
  }
  if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (y < 0.5 || y > (F_HEIGHT+0.5))) {
    fprintf(stderr, "vec2_wrap: Expected y to be in [0.5, %f], got %f\n", F_HEIGHT+0.5, y);
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