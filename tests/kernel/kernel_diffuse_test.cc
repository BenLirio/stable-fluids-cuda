#include <gtest/gtest.h>
#include <gold/diffuse.h>
#include <kernel/diffuse.cuh>
#include <util/macros.h>
#include <cuda_runtime.h>
#include <util/compile_options.h>
#include <math.h>

TEST(Kernel, Diffuse) {
  float *gold_previous_values = (float*)malloc(N*sizeof(float));
  float *gold_values = (float*)malloc(N*sizeof(float));

  float *previous_values = (float*)malloc(N*sizeof(float));
  float *values = (float*)malloc(N*sizeof(float));



  for (int i = 0; i < N; i++) {
    float value = i / (float)N;
    values[i] = value;
    previous_values[i] = value;
    gold_previous_values[i] = value;
    gold_values[i] = value;
  }


  gold_diffuse(gold_previous_values, gold_values, DIFFUSION_RATE);
  kernel_diffuse_wrapper(previous_values, values, DIFFUSION_RATE);


  float total_error = 0.0;
  int number_of_fields_compared = 2;
  for (int i = 0; i < N; i++) {
    total_error += fabs(previous_values[i] - gold_previous_values[i]);
    total_error += fabs(values[i] - gold_values[i]);
  }

  float average_error = total_error/(number_of_fields_compared*N);
  EXPECT_LT(average_error, MAX_AVERAGE_ERROR_THRESHOLD);

  free(previous_values);
  free(values);
  free(gold_previous_values);
  free(gold_values);
}