#include <gold/source_velocities.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/state.h>

void gold_source_velocities() {
  vec2 center = vec2(WIDTH/2.0, HEIGHT/2.0);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2((float)x, (float)y);
      float distance = fmax(vec2_scaled_dist(center, position), 0.01);
      float magnitude = 0.01*((4+(rand()/(float)RAND_MAX))/(distance*distance));
      float percent_complete = (float)current_step / (float)NUM_STEPS;
      float x_magnitude = magnitude*cos(percent_complete*M_PI*10.0);
      float y_magnitude = magnitude*sin(percent_complete*M_PI*10.0);
      x_velocities[IDX2(idx)] += x_magnitude;
      previous_x_velocities[IDX2(idx)] += x_magnitude;
      y_velocities[IDX2(idx)] += y_magnitude;
      previous_y_velocities[IDX2(idx)] += y_magnitude;
    }
  }
}