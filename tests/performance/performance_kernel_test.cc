#include <gtest/gtest.h>
#include <time.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/state.h>
#include <stdio.h>
#include <kernel/index.cuh>

TEST(Performance, NaiveKernel) {
  int current_step = 0;
  state_t state;
  state_malloc(&state);
  state_init(state);

  for (int i = 0; i < NUM_STEPS; i++) {
    kernel_step_wrapper(state, current_step);
    current_step++;
  }

  state_free(state);
}