#include "gold.h"
#include "config.h"
#include "state.h"
#include <stdio.h>

void init() {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      colors[y*WIDTH+x] = x/(float)WIDTH;
      previous_colors[y*WIDTH+x] = x/(float)WIDTH;
      x_velocities[y*WIDTH+x] = 0.01;
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