#include <gtest/gtest.h>

#include <kernel/project.cuh>
#include <gold/project.h>

#include <util/macros.h>
#include <cuda_runtime.h>
#include <util/compile_options.h>
#include <math.h>

TEST(Kernel, Project) {
  size_t number_of_bytes = N*sizeof(float);

  float *x_velocities = (float*)malloc(number_of_bytes);
  float *y_velocities = (float*)malloc(number_of_bytes);
  float *pressures = (float*)malloc(number_of_bytes);
  float *divergences = (float*)malloc(number_of_bytes);

  float *gold_x_velocities = (float*)malloc(number_of_bytes);
  float *gold_y_velocities = (float*)malloc(number_of_bytes);
  float *gold_pressures = (float*)malloc(number_of_bytes);
  float *gold_divergences = (float*)malloc(number_of_bytes);


  for (int y = 1; y <= HEIGHT; y++) {
    float y_velocity = y / (float)HEIGHT;
    for (int x = 1; x <= WIDTH; x++) {
      float x_velocity = x / (float)WIDTH;
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] = x_velocity;
      y_velocities[IDX2(idx)] = y_velocity;
      pressures[IDX2(idx)] = rand() / (float)RAND_MAX;
      divergences[IDX2(idx)] = rand() / (float)RAND_MAX;

      gold_x_velocities[IDX2(idx)] = x_velocity;
      gold_y_velocities[IDX2(idx)] = y_velocity;
      gold_pressures[IDX2(idx)] = rand() / (float)RAND_MAX;
      gold_divergences[IDX2(idx)] = rand() / (float)RAND_MAX;
    }
  }

  float total_error = 0.0;
  int num_fields_compared = 2;
  float max_error = 0.0;
  float average_error;

  for (int i = 0; i < MAX_CONVERGENCE_ITERATIONS; i++) {
    gold_project(gold_x_velocities, gold_y_velocities, gold_pressures, gold_divergences);
    kernel_project_wrapper(x_velocities, y_velocities, pressures, divergences);

    total_error = 0.0;

    for (int i = 0; i < N; i++) {
      float error = 0.0;
      error += fabs(gold_x_velocities[i] - x_velocities[i]);
      error += fabs(gold_y_velocities[i] - y_velocities[i]);

      max_error = max_error > error ? max_error : error;
      total_error += error;
    }

    average_error = total_error / (N * num_fields_compared);
    max_error = max_error / num_fields_compared;
    if (average_error < MAX_AVERAGE_ERROR_THRESHOLD) {
      break;
    }
  }

  EXPECT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);
  EXPECT_LT(max_error, MAX_SINGLE_ERROR_THRESHOLD);

  free(gold_x_velocities);
  free(gold_y_velocities);
  free(gold_pressures);
  free(gold_divergences);
  free(x_velocities);
  free(y_velocities);
  free(pressures);
  free(divergences);
}