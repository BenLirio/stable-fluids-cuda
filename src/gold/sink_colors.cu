#include <gold/sink_colors.cuh>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/macros.h>
#include <util/state.h>

void gold_sink_colors() {
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      colors[IDX2(idx)] *= (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
      previous_colors[IDX2(idx)] *= (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
    }
  }
}