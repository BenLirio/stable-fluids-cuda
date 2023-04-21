#include <gold/source_colors.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/state.h>
#include <stdio.h>
#include <util/type_casting.cuh>

void gold_source_colors(float *previous_colors, float *colors) {
  vec2 center = vec2((WIDTH/2.0)+0.5, (HEIGHT/2.0)+0.5);

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2_of_idx2(idx);
      float distance = vec2_scaled_dist(center, position);
      float magnitude = 1.0 / (distance * distance);
      colors[IDX2(idx)] += 0.1;
      previous_colors[IDX2(idx)] += 0.1;
    }
  }
}