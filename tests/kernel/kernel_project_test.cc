#include <gtest/gtest.h>

#include <kernel/project.cuh>
#include <gold/project.h>

#include <util/macros.h>
#include <cuda_runtime.h>
#include <util/compile_options.h>
#include <math.h>

TEST(Kernel, Project) {
  float *gold_x_velocities = (float*)malloc(N*sizeof(float));
  float *gold_y_velocities = (float*)malloc(N*sizeof(float));
  float *gold_pressures = (float*)malloc(N*sizeof(float));
  float *gold_divergences = (float*)malloc(N*sizeof(float));

  float *x_velocities = (float*)malloc(N*sizeof(float));
  float *y_velocities = (float*)malloc(N*sizeof(float));
  float *pressures = (float*)malloc(N*sizeof(float));
  float *divergences = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    float x_velocity = (i%WIDTH) / (float)WIDTH;
    gold_x_velocities[i] = x_velocity;
    x_velocities[i] = x_velocity;

    float y_velocity = (i/WIDTH) / (float)HEIGHT;
    gold_y_velocities[i] = y_velocity;
    y_velocities[i] = y_velocity;

    float pressure = rand() / (float)RAND_MAX;
    gold_pressures[i] = pressure;
    pressures[i] = pressure;

    float divergence = rand() / (float)RAND_MAX;
    gold_divergences[i] = divergence;
    divergences[i] = divergence;
  }

  gold_project(gold_x_velocities, gold_y_velocities, gold_pressures, gold_divergences);
  kernel_project_wrapper(x_velocities, y_velocities, pressures, divergences);

  float total_error = 0.0;
  for (int i = 0; i < N; i++) {
    total_error += fabs(gold_x_velocities[i] - x_velocities[i]);
    total_error += fabs(gold_y_velocities[i] - y_velocities[i]);
  }

  int num_fields_compared = 2;
  float average_error = total_error/(num_fields_compared*N);
  EXPECT_NEAR(average_error, 0.0, EQ_THRESHOLD);

  free(gold_x_velocities);
  free(gold_y_velocities);
  free(gold_pressures);
  free(gold_divergences);
  free(x_velocities);
  free(y_velocities);
  free(pressures);
  free(divergences);
}