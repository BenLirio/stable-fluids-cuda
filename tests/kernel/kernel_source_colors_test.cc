#include <gtest/gtest.h>

#include <util/macros.h>
#include <kernel/source_colors.cuh>
#include <gold/source_colors.cuh>
#include <math.h>

TEST(Kernel, SourceColors) {
  size_t number_of_bytes = N*sizeof(float);

  float *previous_colors = (float*)malloc(number_of_bytes);
  float *colors = (float*)malloc(number_of_bytes);

  float *gold_colors = (float*)malloc(number_of_bytes);
  float *gold_previous_colors = (float*)malloc(number_of_bytes);

  for (int i = 0; i < N; i++) {
    previous_colors[i] = rand() / (float)RAND_MAX;
    colors[i] = rand() / (float)RAND_MAX;

    gold_previous_colors[i] = previous_colors[i];
    gold_colors[i] = colors[i];
  }

  gold_source_colors(gold_previous_colors, gold_colors);
  kernel_source_colors_wrapper(previous_colors, colors);

  float total_error = 0.0f;
  float max_error = 0.0f;
  int number_of_comparisons = 2;

  for (int i = 0; i < N; i++) {
    float error = 0.0f;
    error += fabs(colors[i] - gold_colors[i]);
    error += fabs(previous_colors[i] - gold_previous_colors[i]);
    total_error += error;
    max_error = max_error > error ? max_error : error;
  }

  max_error = max_error / number_of_comparisons;
  float average_error = total_error / (N * number_of_comparisons);
  ASSERT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);
  ASSERT_LT(max_error, MAX_SINGLE_ERROR_THRESHOLD);


  free(colors);
  free(previous_colors);

  free(gold_colors);
  free(gold_previous_colors);
}