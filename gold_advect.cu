#include "gold_advect.h"
#include "macros.h"
#include "compile_options.h"
#include "vec2.h"
#include "idx2.h"
#include "type_casting.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

void gold_advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = TIME_STEP*sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 pos = vec2((float)x, (float)y);
      vec2 velocity = vec2(x_velocities[IDX2(idx)], y_velocities[IDX2(idx)]);
      vec2 pos_offset_by_velocity = vec2_add(pos, vec2_scale(-alpha, velocity));
      idx2 idx_offset_by_velocity = idx2_of_vec2(pos_offset_by_velocity);
      vec2 pos_offset_by_velocity_floored = vec2_of_idx2(idx_offset_by_velocity);
      float wx0 = min(min(
        abs(pos_offset_by_velocity.x - idx_offset_by_velocity.x),
        abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x + WIDTH))
      ), abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x - WIDTH)));
      float wx1 = 1 - wx0;
      float wy0 = min(min(
        abs(pos_offset_by_velocity.y - idx_offset_by_velocity.y),
        abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y + HEIGHT))
      ), abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y - HEIGHT)));
      float wy1 = 1 - wy0;
      float weights[NUM_NEIGHBORS] = {
        wx1*wy1,
        wx1*wy0,
        wx0*wy1,
        wx0*wy0,
      };
      values[IDX2(idx)] = 0;
      for (int i = 0; i < NUM_NEIGHBORS; i++) {
        float weight = weights[i];
        idx2 neighbor = idx2_wrap(idx2(
          idx_offset_by_velocity.x + lower_right_square_offsets[i].x,
          idx_offset_by_velocity.y + lower_right_square_offsets[i].y
        ));
        values[IDX2(idx)] += weight*previous_values[IDX2(neighbor)];
      }
    }
  }
}
