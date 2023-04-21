#include <gold/sink_velocities.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/state.h>

void gold_sink_velocities(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) {
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      previous_x_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      y_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      previous_y_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
    }
  }
}
