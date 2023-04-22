#include <gold/advect.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/type_casting.cuh>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

void gold_advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = TIME_STEP*sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 pos = vec2_of_idx2(idx);
      vec2 velocity = vec2(x_velocities[IDX2(idx)], y_velocities[IDX2(idx)]);
      vec2 pos_offset_by_velocity = vec2_add(pos, vec2_scale(-alpha, velocity));
      idx2 idx_offset_by_velocity = idx2_of_vec2(pos_offset_by_velocity);
      vec2 pos_offset_by_velocity_floored = vec2_of_idx2(idx_offset_by_velocity);
      float wx0 = vec2_x_dist(pos_offset_by_velocity, pos_offset_by_velocity_floored);
      float wx1 = 1 - wx0;
      float wy0 = vec2_y_dist(pos_offset_by_velocity, pos_offset_by_velocity_floored);
      float wy1 = 1 - wy0;
      values[IDX2(idx)] = (
          wx1*wy1*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(0, 0)))]
        + wx1*wy0*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(0, 1)))]
        + wx0*wy1*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(1, 0)))]
        + wx0*wy0*previous_values[IDX2(idx2_add(idx_offset_by_velocity, idx2(1, 1)))]
      );
    }
  }
}
