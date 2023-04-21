#include <gold/source_velocities.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/state.h>
#include <util/type_casting.cuh>

void gold_source_velocities(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities, int current_step) {
  vec2 center = vec2((WIDTH/2.0)+0.5, (HEIGHT/2.0)+0.5);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2_of_idx2(idx);

      float distance = vec2_scaled_dist(center, position);
      float magnitude = 1/(distance*distance);
      float percent_complete = (float)current_step / (float)NUM_STEPS;
      float x_magnitude = magnitude*cos(percent_complete*M_PI*10.0);
      float y_magnitude = magnitude*sin(percent_complete*M_PI*10.0);

      x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP;
      previous_x_velocities[IDX2(idx)] += x_magnitude*TIME_STEP;
      y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP;
      previous_y_velocities[IDX2(idx)] += y_magnitude*TIME_STEP;
    }
  }
}