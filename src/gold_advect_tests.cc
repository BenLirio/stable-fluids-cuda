#include <gtest/gtest.h>
#include "gold_advect.h"
#include <macros.h>

#define NUM_ADVECTION_STEPS 100
TEST(Advect, zero_sum) {
  float *x_velocities = (float*)malloc(N * sizeof(float));
  float *y_velocities = (float*)malloc(N * sizeof(float));
  float *previous_color = (float*)malloc(N * sizeof(float));
  float *color = (float*)malloc(N * sizeof(float));

  float original_total = 0.0;
  for (int i = 0; i < N; i++) {
    color[i] = rand() / (float)RAND_MAX;
    previous_color[i] = color[i];
    original_total += color[i];
    x_velocities[i] = 1.23142323;
    y_velocities[i] = -13.12324123;
  }

  for (int i = 0; i < NUM_ADVECTION_STEPS; i++) {
    SWAP(previous_color, color);
    gold_advect(previous_color, color, x_velocities, y_velocities);
  }

  float total = 0.0;
  for (int i = 0; i < N; i++) {
    total += color[i];
  }

  float difference = total - original_total;
  float scaled_difference = difference / original_total;

  EXPECT_NEAR(scaled_difference, 0.0, EQ_THRESHOLD);


  free(x_velocities);
  free(y_velocities);
  free(previous_color);
  free(color);
}