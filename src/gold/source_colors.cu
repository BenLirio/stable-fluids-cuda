#include <gold/source_colors.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/state.h>

void gold_source_colors() {
  vec2 center = vec2(WIDTH/2.0, HEIGHT/2.0);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2((float)x, (float)y);
      float distance = fmax(vec2_scaled_dist(center, position), 0.01);
      float magnitude = 0.01*((4+(rand()/(float)RAND_MAX))/(distance*distance));
      colors[IDX2(idx)] += magnitude;
      previous_colors[IDX2(idx)] += magnitude;
    }
  }
}