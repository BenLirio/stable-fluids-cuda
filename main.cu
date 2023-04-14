#include "gold.h"
#include "compile_options.h"
#include "macros.h"
#include "state.h"
#include "idx2.h"
#include <stdio.h>
#include "vec2.h"

float _vxs0[N];
float _vxs[N];
float _vys0[N];
float _vys[N];
float _cs0[N];
float _cs[N];
float _preasure[N];
float _divergence[N];

void init() {
  previous_x_velocities = _vxs0;
  x_velocities = _vxs;
  previous_y_velocities = _vys0;
  y_velocities = _vys;
  previous_colors = _cs0;
  colors = _cs;
  preasure = _preasure;
  divergence = _divergence;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      colors[IDX2(idx)] = rand()/((float)RAND_MAX);
      x_velocities[IDX2(idx)] = 0.0;
      previous_x_velocities[IDX2(idx)] = 0.0;
      y_velocities[IDX2(idx)] = 0.0;
      previous_y_velocities[IDX2(idx)] = 0.0;
    }
  }
}

int main() {
  init();
  for (int i = 0; i < NUM_STEPS; i++) {
    if (i != 0)
      printf(",");
    for (int y = 0; y < HEIGHT; y++) {
      for (int x = 0; x < WIDTH; x++) {
        printf("%f", colors[y*WIDTH+x]);
        if (y != HEIGHT - 1 || x != WIDTH - 1)
          printf(",");
      }
    }
    step();
  }
  return 0;
}