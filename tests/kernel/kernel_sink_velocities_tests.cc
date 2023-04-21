#include <gtest/gtest.h>
#include <util/macros.h>
#include <kernel/sink_velocities.cuh>
#include <gold/sink_velocities.cuh>
#include <math.h>

TEST(Kernel, SinkVelocities) {
  size_t number_of_bytes = N*sizeof(float);

  float *previous_x_velocities = (float*)malloc(number_of_bytes);
  float *previous_y_velocities = (float*)malloc(number_of_bytes);
  float *x_velocities = (float*)malloc(number_of_bytes);
  float *y_velocities = (float*)malloc(number_of_bytes);

  float *gold_previous_x_velocities = (float*)malloc(number_of_bytes);
  float *gold_previous_y_velocities = (float*)malloc(number_of_bytes);
  float *gold_x_velocities = (float*)malloc(number_of_bytes);
  float *gold_y_velocities = (float*)malloc(number_of_bytes);


  for (int i = 0; i < N; i++) {
    previous_x_velocities[i] = rand() / (float)RAND_MAX;
    previous_y_velocities[i] = rand() / (float)RAND_MAX;
    x_velocities[i] = rand() / (float)RAND_MAX;
    y_velocities[i] = rand() / (float)RAND_MAX;

    gold_previous_x_velocities[i] = previous_x_velocities[i];
    gold_previous_y_velocities[i] = previous_y_velocities[i];
    gold_x_velocities[i] = x_velocities[i];
    gold_y_velocities[i] = y_velocities[i];
  }

  gold_sink_velocities(gold_previous_x_velocities, gold_previous_y_velocities, gold_x_velocities, gold_y_velocities);
  kernel_sink_velocities_wrapper(previous_x_velocities, previous_y_velocities, x_velocities, y_velocities);

  float total_error = 0.0f;
  float max_error = 0.0f;
  int number_of_comparisons = 4;

  for (int i = 0; i < N; i++) {
    float error = 0.0f;
    error += fabs(gold_previous_x_velocities[i] - previous_x_velocities[i]);
    error += fabs(gold_previous_y_velocities[i] - previous_y_velocities[i]);
    error += fabs(gold_x_velocities[i] - x_velocities[i]);
    error += fabs(gold_y_velocities[i] - y_velocities[i]);
    total_error += error;
    max_error = max_error > error ? max_error : error;
  }

  max_error = max_error / number_of_comparisons;
  float average_error = total_error / (N * number_of_comparisons);
  ASSERT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);
  ASSERT_LT(max_error, MAX_SINGLE_ERROR_THRESHOLD);


  free(previous_x_velocities);
  free(previous_y_velocities);
  free(x_velocities);
  free(y_velocities);
  
  free(gold_previous_x_velocities);
  free(gold_previous_y_velocities);
  free(gold_x_velocities);
  free(gold_y_velocities);
}