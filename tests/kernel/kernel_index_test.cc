#include <gtest/gtest.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <gold/index.h>
#include <kernel/index.cuh>
#include <math.h>
#include <util/state.h>

TEST(Kernel, Index) {

  state_t state;
  state_malloc(&state);
  state_init(state);

  state_t gold_state;
  state_malloc(&gold_state);
  state_init(gold_state);

  int current_step = 0;

  kernel_step_wrapper(state, current_step);
  gold_step(gold_state, current_step);

  float total_error = 0.0;
  float max_error = 0.0;
  int number_of_comparisons = 3;
  int x_loc;
  int y_loc;

  for (int i = 0; i < N; i++) {
    float error = 0.0;
    error += fabs(state.colors->current[i] - gold_state.colors->current[i]);
    error += fabs(state.x_velocities->current[i] - gold_state.x_velocities->current[i]);
    error += fabs(state.y_velocities->current[i] - gold_state.y_velocities->current[i]);
    total_error += error;
    if (error > max_error) {
      x_loc = i % WIDTH;
      y_loc = i / WIDTH;
      printf("error %f at (%d, %d)\n", error, x_loc, y_loc);
    }
    max_error = max_error > error ? max_error : error;
  }

  float average_error = total_error / (N * number_of_comparisons);
  max_error = max_error / number_of_comparisons;

  EXPECT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);
  EXPECT_LT(max_error, MAX_SINGLE_ERROR_THRESHOLD);

  state_free(state);
  state_free(gold_state);
}