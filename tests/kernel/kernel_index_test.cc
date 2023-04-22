#include <gtest/gtest.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <gold/index.h>
#include <kernel/index.cuh>
#include <math.h>

TEST(Kernel, Index) {
  float *colors = (float*)malloc(N*sizeof(float));
  float *previous_colors = (float*)malloc(N*sizeof(float));
  float *previous_x_velocities = (float*)malloc(N*sizeof(float));
  float *previous_y_velocities = (float*)malloc(N*sizeof(float));
  float *x_velocities = (float*)malloc(N*sizeof(float));
  float *y_velocities = (float*)malloc(N*sizeof(float));
  float *pressures = (float*)malloc(N*sizeof(float));
  float *divergences = (float*)malloc(N*sizeof(float));

  float *gold_colors = (float*)malloc(N*sizeof(float));
  float *gold_previous_colors = (float*)malloc(N*sizeof(float));
  float *gold_previous_x_velocities = (float*)malloc(N*sizeof(float));
  float *gold_previous_y_velocities = (float*)malloc(N*sizeof(float));
  float *gold_x_velocities = (float*)malloc(N*sizeof(float));
  float *gold_y_velocities = (float*)malloc(N*sizeof(float));
  float *gold_pressures = (float*)malloc(N*sizeof(float));
  float *gold_divergences = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    colors[i] = 0;
    previous_colors[i] = 0;
    previous_x_velocities[i] = 0;
    previous_y_velocities[i] = 0;
    x_velocities[i] = 0;
    y_velocities[i] = 0;
    pressures[i] = 0;
    divergences[i] = 0;

    gold_colors[i] = 0;
    gold_previous_colors[i] = 0;
    gold_previous_x_velocities[i] = 0;  
    gold_previous_y_velocities[i] = 0;
    gold_x_velocities[i] = 0;
    gold_y_velocities[i] = 0;
    gold_pressures[i] = 0;
    gold_divergences[i] = 0;
  }

  int current_step = 0;

  gold_step(
    gold_colors,
    gold_previous_colors,
    gold_previous_x_velocities,
    gold_previous_y_velocities,
    gold_x_velocities,
    gold_y_velocities,
    gold_pressures,
    gold_divergences,
    current_step
  );
  kernel_step_wrapper(
    colors,
    previous_colors,
    previous_x_velocities,
    previous_y_velocities,
    x_velocities,
    y_velocities,
    pressures,
    divergences,
    current_step
  );

  float total_error = 0.0;
  float max_error = 0.0;
  int number_of_comparisons = 3;

  for (int i = 0; i < N; i++) {
    float error = 0.0;
    error += fabs(gold_colors[i] - colors[i]);
    error += fabs(gold_x_velocities[i] - x_velocities[i]);
    error += fabs(gold_y_velocities[i] - y_velocities[i]);

    total_error += error;
    max_error = max_error > error ? max_error : error;
  }

  float average_error = total_error / (N * number_of_comparisons);
  max_error = max_error / number_of_comparisons;

  EXPECT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);
  EXPECT_LT(max_error, MAX_SINGLE_ERROR_THRESHOLD);

  free(colors);
  free(previous_colors);
  free(previous_x_velocities);
  free(previous_y_velocities);
  free(x_velocities);
  free(y_velocities);
  free(pressures);
  free(divergences);

  free(gold_colors);
  free(gold_previous_colors);
  free(gold_previous_x_velocities);
  free(gold_previous_y_velocities);
  free(gold_x_velocities);
  free(gold_y_velocities);
  free(gold_pressures);
  free(gold_divergences);
}