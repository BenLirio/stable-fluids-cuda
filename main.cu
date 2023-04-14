#include "gold.h"
#include "config.h"
#include "state.h"
#include "idx2.h"
#include <stdio.h>
#include "vec2.h"

void init() {
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