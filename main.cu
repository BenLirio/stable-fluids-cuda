#include "gold.h"
#include "config.h"
#include "state.h"
#include <stdio.h>
#include "vec2.h"

void init() {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      idx2 idx = idx2(x, y);
      colors[IDX2(idx)] = rand()/((float)RAND_MAX);
      x_velocities[IDX2(idx)] = 0.0;
      previous_x_velocities[IDX2(idx)] = 0.0;
      y_velocities[IDX2(idx)] = 0.0;
      previous_y_velocities[IDX2(idx)] = 0.0;
      // float magnitude = 0;
      // if (y < HEIGHT/10.0 && y > HEIGHT/20.0) {
      //   magnitude = 1.0;
      // }
      // if (y > HEIGHT - (HEIGHT/10.0) && y < HEIGHT - (HEIGHT/20.0)) {
      //   magnitude = -1.0;
      // }
      // float sin_value = sin(((x+y)/(WIDTH+HEIGHT))*M_PI_2);
      // float cos_value = cos(((x+y)/(WIDTH+HEIGHT))*M_PI_2);
      // y_velocities[IDX2(idx)] = 100*(magnitude * (1 + fabs(sin_value) + 0.1*(rand()/((float)RAND_MAX))));
      // previous_y_velocities[IDX2(idx)] = y_velocities[IDX2(idx)];
      // x_velocities[IDX2(idx)] = 0.0;
      // colors[IDX2(idx)] = fabs(magnitude) * (fabs(cos_value) + 0.1*(rand()/((float)RAND_MAX)));
      // previous_colors[IDX2(idx)] = colors[IDX2(idx)];
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