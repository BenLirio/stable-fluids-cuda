#include <gold/index.h>
#include <util/compile_options.h>
#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>

void output_color(float *colors, int i) {
  if (i != 0)
    printf(",");
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      printf("%f", colors[y*WIDTH+x]);
      if (y != HEIGHT - 1 || x != WIDTH - 1)
        printf(",");
    }
  }
}

int main() {
  int current_step = 0;
  state_t state;
  state_malloc(&state);
  state_init(state);
  void (*step)(state_t, int) = USE_GOLD ? gold_step : kernel_step_wrapper;

  for (int i = 0; i < NUM_STEPS; i++) {
    output_color(state.colors->current, current_step);
    step(state, current_step);
    current_step++;
  }

  state_free(state);
  return 0;
}