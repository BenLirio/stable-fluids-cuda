#include <gtest/gtest.h>
#include <omp.h>
#include "idx2.h"
#include "vec2.h"
#include "gold.h"
#include "compile_options.h"
#include <math.h>
#include "macros.h"
#include "gold_diffuse.h"
#include "math.h"
#include <stdlib.h>

#define STEPS_UNTIL_STABLE 700
#define MAX_N_TO_TEST (128*128)

TEST(Diffuse, disperses_evenly) {
  if (N > MAX_N_TO_TEST) {
    EXPECT_TRUE(false);
    return;
  }
  float *previous_color = (float*)malloc(N * sizeof(float));
  float *color = (float*)malloc(N * sizeof(float));

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    previous_color[i] = (float)i/(float)N;
    color[i] = (float)i/(float)N;
  }

  for (int i = 0; i < STEPS_UNTIL_STABLE; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  float total = 0.0;
  #pragma omp parallel for reduction(+:total)
  for (int i = 0; i < N; i++) {
    total += color[i];
  }

  float average = total / (float)N;
  float total_error = 0.0;
  for (int i = 0; i < N; i++) {
    total_error += fabs(color[i] - average);
  }
  float average_error = total_error / (float)N;
  EXPECT_NEAR(average_error, 0.0, EQ_THRESHOLD);

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

  for (int i = 0; i < STEPS_UNTIL_STABLE; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  float total = 0.0;
  #pragma omp parallel for reduction(+:total)
  for (int i = 0; i < N; i++) {
    total += color[i];
  }

  float difference = original_total - total;
  float scaled_difference = difference / original_total;

  EXPECT_NEAR(scaled_difference, 0.0, EQ_THRESHOLD);

  free(previous_color);
  free(color);
}