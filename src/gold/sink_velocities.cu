#include <gold/sink_velocities.cuh>

#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/state.h>

void gold_sink_velocities(float *previous_x_velocities, float *previous_y_velocities, float *x_velocities, float *y_velocities) {
  float alpha = (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] *= alpha;
      previous_x_velocities[IDX2(idx)] *= alpha;
      y_velocities[IDX2(idx)] *= alpha;
      previous_y_velocities[IDX2(idx)] *= alpha;
    }
  }
}
