#include <gtest/gtest.h>

#include <kernel/advect.cuh>
#include <gold/advect.h>

#include <util/macros.h>
#include <cuda_runtime.h>
#include <util/compile_options.h>
#include <math.h>

TEST(Kernel, Advect) {
  float *gold_previous_values = (float*)malloc(N*sizeof(float));
  float *gold_values = (float*)malloc(N*sizeof(float));
  float *gold_x_velocities = (float*)malloc(N*sizeof(float));
  float *gold_y_velocities = (float*)malloc(N*sizeof(float));


  float *previous_values = (float*)malloc(N*sizeof(float));
  float *values = (float*)malloc(N*sizeof(float));
  float *x_velocities = (float*)malloc(N*sizeof(float));
  float *y_velocities = (float*)malloc(N*sizeof(float));


  for (int i = 0; i < N; i++) {
    float value = i / (float)N;
    values[i] = value;
    previous_values[i] = value;
    gold_previous_values[i] = value;
    gold_values[i] = value;

    float y_velocity = (i/WIDTH) / (float)HEIGHT;
    y_velocities[i] = y_velocity;
    gold_y_velocities[i] = y_velocity;

    float x_velocity = (i%WIDTH) / (float)WIDTH;
    x_velocities[i] = x_velocity;
    gold_x_velocities[i] = x_velocity;
  }

  gold_advect(gold_previous_values, gold_values, gold_x_velocities, gold_y_velocities);
  kernel_advect_wrapper(previous_values, values, x_velocities, y_velocities);
  


  float total_error = 0.0;
  for (int i = 0; i < N; i++) {
    total_error += fabs(previous_values[i] - gold_previous_values[i]);
    total_error += fabs(values[i] - gold_values[i]);
    total_error += fabs(x_velocities[i] - gold_x_velocities[i]);
    total_error += fabs(y_velocities[i] - gold_y_velocities[i]);
  }

  float average_error = total_error/(4*N);
  EXPECT_NEAR(average_error, 0.0, EQ_THRESHOLD);

  free(previous_values);
  free(values);
  free(gold_previous_values);
  free(gold_values);
  free(x_velocities);
  free(y_velocities);
  free(gold_x_velocities);
  free(gold_y_velocities);
}