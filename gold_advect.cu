#include "gold_advect.h"
#include "macros.h"
#include "compile_options.h"
#include "vec2.h"
#include "idx2.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

idx2 lower_right_square_offsets[NUM_NEIGHBORS] = {
  idx2(0, 0),
  idx2(0, 1),
  idx2(1, 0),
  idx2(1, 1),
};

void gold_advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = TIME_STEP*sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 pos_offset_by_velocity = wrap_vec2(vec2(
        x - alpha*x_velocities[IDX2(idx)],
        y - alpha*y_velocities[IDX2(idx)]
      ));
      idx2 idx_offset_by_velocity = wrap_idx2(idx2(
        (int)floor(pos_offset_by_velocity.x),
        (int)floor(pos_offset_by_velocity.y)
      ));
      float wx0 = min(min(
        abs(pos_offset_by_velocity.x - idx_offset_by_velocity.x),
        abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x + WIDTH))
      ), abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x - WIDTH)));
      if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (wx0 < 0.0f || wx0 > 1.0f)) {
        fprintf(stderr, "advect: y=%d; x=%d; pos=%f; idx=%d; wx0=%f;\n", y, x, pos_offset_by_velocity.x, idx_offset_by_velocity.x, wx0);
      }
      if (ASSERTIONS_ENABLED) assert(wx0 >= 0.0f && wx0 <= 1.0f);
      float wx1 = 1 - wx0;
      if (ASSERTIONS_ENABLED) assert(wx1 >= 0.0f && wx1 <= 1.0f);
      float wy0 = min(min(
        abs(pos_offset_by_velocity.y - idx_offset_by_velocity.y),
        abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y + HEIGHT))
      ), abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y - HEIGHT)));
      if (ASSERTIONS_ENABLED) assert(wy0 >= 0.0f && wy0 <= 1.0f);
      float wy1 = 1 - wy0;
      if (ASSERTIONS_ENABLED) assert(wy1 >= 0.0f && wy1 <= 1.0f);
      float weights[NUM_NEIGHBORS] = {
        wx1*wy1,
        wx1*wy0,
        wx0*wy1,
        wx0*wy0,
      };
      values[IDX2(idx)] = 0;
      for (int i = 0; i < NUM_NEIGHBORS; i++) {
        float weight = weights[i];
        idx2 neighbor = wrap_idx2(idx2(
          idx_offset_by_velocity.x + lower_right_square_offsets[i].x,
          idx_offset_by_velocity.y + lower_right_square_offsets[i].y
        ));
        values[IDX2(idx)] += weight*previous_values[IDX2(neighbor)];
      }
    }
  }
}
