#include <gtest/gtest.h>
#include <util/macros.h>
#include <gold/project.h>
#include <util/idx2.h>
#include <math.h>

TEST(PROJECT, reduce_gradient) {
  float *x_velocities = (float*)malloc(N*sizeof(float));
  float *y_velocities = (float*)malloc(N*sizeof(float));
  float *pressure = (float*)malloc(N*sizeof(float));
  float *divergence = (float*)malloc(N*sizeof(float));

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] = (y/(float)HEIGHT) + (x/(float)WIDTH);
      y_velocities[IDX2(idx)] = (y/(float)HEIGHT) + (x/(float)WIDTH);
      // pressure and divergence overwritten, but set here
      // to make sure they are no assumptions
      pressure[IDX2(idx)] = rand() / (float)RAND_MAX;
      divergence[IDX2(idx)] = rand() / (float)RAND_MAX;
    }
  }
  float original_total_gradient = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      original_total_gradient += fabs(get_x_derivative(x_velocities, idx) + get_y_derivative(y_velocities, idx));
    }
  }
  float original_scaled_gradient = original_total_gradient / N;

  gold_project(x_velocities, y_velocities, pressure, divergence);

  float total_gradient = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      total_gradient += fabs(get_x_derivative(x_velocities, idx) + get_y_derivative(y_velocities, idx));
    }
  }
  float total_scaled_gradient = total_gradient / N;
  EXPECT_LT(total_scaled_gradient, original_total_gradient);

  free(x_velocities);
  free(y_velocities);
  free(divergence);
  free(pressure);
}