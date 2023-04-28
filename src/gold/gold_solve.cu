#include <gold/solve.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>
#include <util/performance.cuh>
#include <stdio.h>
#include <math.h>

void gold_solve_loop_body(
    int x,
    int y,
    float *base,
    float *values,
    float factor,
    float divisor,
    float *expected_x,
    bool print_error,
    float *delta,
    float *error
) {
  idx2 idx = idx2(x, y);
  float x_prime = (
    base[IDX2(idx)] +
    factor * (
      values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(0, -1)))] +
      values[IDX2(idx2_add(idx, idx2(0, 1)))]
    )
  ) / divisor;
  *delta += fabs(x_prime - values[IDX2(idx)]);
  if (print_error)
    *error += fabs(x_prime - expected_x[IDX2(idx)]);
  values[IDX2(idx)] = x_prime;
}

int gold_solve(float *base, float *values, float factor, float divisor, float *expected_x, bool print_error) {
  for (int num_iterations = 1; ; num_iterations++) {
    float error = 0.0;
    float delta = 0.0;
    for (int red_black = 0; red_black < 2; red_black++) {
      for (int y = 1; y <= HEIGHT; y++) {
        for (int x = 1; x <= WIDTH; x++) {
          if ((x%2) == ((y+red_black+1)%2)) continue;
          gold_solve_loop_body(
            x, y, base, values, factor, divisor, expected_x, print_error,
            &delta, &error);
        }
      }
    }
    float average_delta = delta / N;
    float average_error = error / N;
    if (print_error && OUTPUT_PERFORMANCE) {
      print_tags(SOLVE_TAG);
      printf("[step=%d][error=%f]\n", num_iterations, average_error);
    }
    if (average_delta < GUASS_SEIDEL_THREASHOLD || num_iterations == MAX_CONVERGENCE_ITERATIONS)
      return num_iterations;
  }
}