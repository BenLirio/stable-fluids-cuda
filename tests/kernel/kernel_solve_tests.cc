#include <gtest/gtest.h>
#include <gold/solve.cuh>
#include <kernel/solve.cuh>
#include <util/macros.h>
#include <cmath>

// TEST(Kernel, Solve) {
//   float *bs = (float*)malloc(N*sizeof(float));
//   float *xs = (float*)malloc(N*sizeof(float));
//   float *d_bs, *d_xs;
//   cudaMalloc(&d_bs, N*sizeof(float));
//   cudaMalloc(&d_xs, N*sizeof(float));
//   float *g_bs = (float*)malloc(N*sizeof(float));
//   float *g_xs = (float*)malloc(N*sizeof(float));

//   float factor = TIME_STEP*DIFFUSION_RATE*N;
//   float divisor = 1.0f + 4*factor;

//   free(bs);
//   free(xs);
//   cudaFree(d_bs);
//   cudaFree(d_xs);
//   free(g_bs);
//   free(g_xs);
// }
// 
// TEST(Kernel, Solve) {
  // float *base = (float*)malloc(N*sizeof(float));
  // float *values = (float*)malloc(N*sizeof(float));
  // float *gold_base = (float*)malloc(N*sizeof(float));
  // float *gold_values = (float*)malloc(N*sizeof(float));
  // for (int i = 0; i < N; i++) {
    // gold_base[i] = rand() / (float)RAND_MAX;
    // gold_values[i] = rand() / (float)RAND_MAX;
    // base[i] = gold_base[i];
    // values[i] = gold_values[i];
  // }
// 
  // float factor = TIME_STEP*DIFFUSION_RATE*N;
  // float divisor = 1.0f + 4*factor;
// 
  // gold_solve_gauss_seidel(gold_base, gold_values, factor, divisor);
// 
  // float *device_base;
  // float *device_values;
  // float *device_expected_values;
  // cudaMalloc(&device_base, N*sizeof(float));
  // cudaMalloc(&device_values, N*sizeof(float));
  // cudaMalloc(&device_expected_values, N*sizeof(float));
  // cudaMemcpy(device_base, base, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(device_values, values, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(device_expected_values, gold_values, N*sizeof(float), cudaMemcpyHostToDevice);
  // int num_iterations = kernel_solve(0, device_base, device_values, device_expected_values, factor, divisor, 0);
  // EXPECT_NE(num_iterations, MAX_CONVERGENCE_ITERATIONS);
  // cudaMemcpy(values, device_values, N*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaFree(device_base);
  // cudaFree(device_values);
// 
  // float total_error = 0.0f;
  // for (int i = 0; i < N; i++) {
    // total_error += std::fabs(values[i] - gold_values[i]);
  // }
  // EXPECT_LT(total_error/N, EQ_THRESHOLD);
// }