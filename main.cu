#include "gold.h"
#include "config.h"
#include "state.h"
#include <stdio.h>

void init() {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // colors[y*WIDTH+x] = (rand()/((float)RAND_MAX*4)) + ((x+y)/(float)(WIDTH+HEIGHT));
      // previous_colors[y*WIDTH+x] = colors[y*WIDTH+x];
      // y_velocities[y*WIDTH+x] = 1 + 5*(rand()/((float)RAND_MAX*5));
      // previous_y_velocities[y*WIDTH+x] = y_velocities[y*WIDTH+x];
      // x_velocities[y*WIDTH+x] = 1 + 5*(rand()/((float)RAND_MAX*5));
      // previous_x_velocities[y*WIDTH+x] = x_velocities[y*WIDTH+x];
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