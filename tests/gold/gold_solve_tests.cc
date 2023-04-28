#include <gtest/gtest.h>
#include <gold/solve.cuh>
#include <util/macros.h>
#include <cmath>


TEST(Gold, Solve) {
  float *b = (float*)malloc(N*sizeof(float));
  float *x = (float*)malloc(N*sizeof(float));
  float *expected_x = (float*)malloc(N*sizeof(float));

  float factor = TIME_STEP*DIFFUSION_RATE*N;
  float divisor = 1.0f + 4*factor;

  bool print_error = false;
  float total_error;

  for (int i = 0; i < N; i++) {
    b[i] = rand() / (float)RAND_MAX;
  }

  for (int i = 0; i < N; i++) {
    x[i] = rand() / (float)RAND_MAX;
  }

  // First pass
  int num_iterations = gold_solve(b, x, factor, divisor, expected_x, print_error);

  EXPECT_NE(num_iterations, MAX_CONVERGENCE_ITERATIONS);

  for (int i = 0; i < N; i++) {
    expected_x[i] = x[i];
  }

  for (int i = 0; i < N; i++) {
    x[i] = rand() / (float)RAND_MAX;
  }

  // Second pass
  print_error = true;
  gold_solve(b, x, factor, divisor, expected_x, print_error);

  total_error = 0.0f;
  for (int i = 0; i < N; i++) {
    total_error += std::fabs(expected_x[i] - x[i]);
  }
  ASSERT_LT(total_error/N, EQ_THRESHOLD);


  free(b);
  free(x);
  free(expected_x);
}