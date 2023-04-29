#include <gtest/gtest.h>
#include <gold/solve.cuh>
#include <util/macros.h>
#include <cmath>
#include <util/idx2.cuh>


TEST(Gold, Solve0) {
  float *bs = (float*)malloc(N*sizeof(float));
  float *xs = (float*)malloc(N*sizeof(float));

  float factor = TIME_STEP*DIFFUSION_RATE*N;
  float divisor = 1.0f + 4*factor;

  for (int i = 0; i < N; i++) {
    bs[i] = rand() / (float)RAND_MAX;
  }

  for (int i = 0; i < N; i++) {
    xs[i] = rand() / (float)RAND_MAX;
  }

  EXPECT_NE(
    gold_solve_conjugate_gradient(bs, xs, factor, divisor),
    MAX_CONVERGENCE_ITERATIONS
  );

  float total_error = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      if (xs[IDX2(idx)] != xs[IDX2(idx)]) {
        ASSERT_TRUE(false);
      }
      total_error += std::fabs(
        bs[IDX2(idx)]
        - (
          + divisor*xs[IDX2(idx)]
          - factor*xs[IDX2(idx2_add(idx, idx2(1, 0)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(-1, 0)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(0, 1)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(0, -1)))]
        )
      );
    }
  }

  ASSERT_LT(total_error/N, EQ_THRESHOLD);

  free(bs);
  free(xs);
}

TEST(Gold, Solve1) {
  float *bs = (float*)malloc(N*sizeof(float));
  float *div = (float*)malloc(N*sizeof(float));
  float *xs = (float*)malloc(N*sizeof(float));

  float factor = 1.0f;
  float divisor = 4.0f;

  for (int i = 0; i < N; i++) {
    bs[i] = rand() / (float)RAND_MAX;
  }
  float h = 1.0f / (WIDTH + 1);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      div[IDX2(idx)] = -0.5f * h * (
        + (bs[IDX2(idx2_add(idx, idx2(1, 0)))] - bs[IDX2(idx2_add(idx, idx2(-1, 0)))])
        + (bs[IDX2(idx2_add(idx, idx2(0, 1)))] - bs[IDX2(idx2_add(idx, idx2(0, -1)))])
      );
      xs[IDX2(idx)] = 0.0f;
    }
  }

  EXPECT_NE(
    gold_solve_conjugate_gradient(div, xs, factor, divisor),
    MAX_CONVERGENCE_ITERATIONS
  );

  float total_error = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      if (xs[IDX2(idx)] != xs[IDX2(idx)]) {
        ASSERT_TRUE(false);
      }
      total_error += std::fabs(
        div[IDX2(idx)]
        - (
          + divisor*xs[IDX2(idx)]
          - factor*xs[IDX2(idx2_add(idx, idx2(1, 0)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(-1, 0)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(0, 1)))]
          - factor*xs[IDX2(idx2_add(idx, idx2(0, -1)))]
        )
      );
    }
  }

  ASSERT_LT(total_error/N, EQ_THRESHOLD);

  free(bs);
  free(xs);
}
