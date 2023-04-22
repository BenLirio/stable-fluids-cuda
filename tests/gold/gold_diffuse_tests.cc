#include <gtest/gtest.h>
#include <omp.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <gold/index.h>
#include <util/compile_options.h>
#include <math.h>
#include <util/macros.h>
#include <gold/diffuse.h>
#include "math.h"
#include <stdlib.h>

#define MAX_N_TO_TEST (128*128)

TEST(Diffuse, disperses_evenly) {
  if (N > MAX_N_TO_TEST) {
    EXPECT_TRUE(false);
    return;
  }
  float *previous_color = (float*)malloc(N * sizeof(float));
  float *color = (float*)malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    previous_color[i] = (float)i/(float)N;
    color[i] = (float)i/(float)N;
  }

  float total_error;

  for (int i = 0; i < MAX_CONVERGENCE_ITERATIONS; i++) {
    float *temp = previous_color;
    previous_color = color;
    color = temp;
    gold_diffuse(previous_color, color, DIFFUSION_RATE);

    if (i % CHECK_CONVERGENCE_EVERY != 0) {
      continue;
    }
    float total = 0.0;
    for (int i = 0; i < N; i++) {
      total += color[i];
    }
    float average = total / (float)N;
    total_error = 0.0;
    for (int i = 0; i < N; i++) {
      total_error += fabs(color[i] - average);
    }
    total_error = total_error / (float)N;
      if (total_error < EQ_THRESHOLD){
      break;
    }
  }

  EXPECT_LT(total_error, EQ_THRESHOLD);

  free(previous_color);
  free(color);
}

TEST(Diffuse, zero_sum) {
  if (N > MAX_N_TO_TEST) {
    EXPECT_TRUE(false);
    return;
  }
  float *previous_color = (float*)malloc(N * sizeof(float));
  float *color = (float*)malloc(N * sizeof(float));

  float original_total = 0.0;
  for (int i = 0; i < N; i++) {
    previous_color[i] = (float)i/(float)N;
    color[i] = previous_color[i];
    original_total += color[i];
  }

  for (int i = 0; i < MAX_CONVERGENCE_ITERATIONS; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  float total = 0.0;
  for (int i = 0; i < N; i++) {
    total += color[i];
  }

  float difference = original_total - total;
  float scaled_difference = difference / original_total;

  EXPECT_NEAR(scaled_difference, 0.0, EQ_THRESHOLD);

  free(previous_color);
  free(color);
}