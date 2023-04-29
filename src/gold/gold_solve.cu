#include <gold/solve.cuh>
#include <util/macros.h>
#include <util/idx2.cuh>
#include <util/performance.cuh>
#include <stdio.h>
#include <math.h>
#include <omp.h>

float gold_solve_gauss_seidel_loop_body(
    int x,
    int y,
    float *base,
    float *values,
    float factor,
    float divisor
) {
  idx2 idx = idx2(x, y);
  float x_prime = (
    base[IDX2(idx)] +
    factor * (
      values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(1, 0)))] +
      values[IDX2(idx2_add(idx, idx2(0, -1)))] +
      values[IDX2(idx2_add(idx, idx2(0, 1)))]
    )
  ) / divisor;
  float delta = fabs(x_prime - values[IDX2(idx)]);
  values[IDX2(idx)] = x_prime;
  return delta;
}

int gold_solve_gauss_seidel(float *base, float *values, float factor, float divisor) {
  for (int num_iterations = 1; num_iterations <= MAX_CONVERGENCE_ITERATIONS; num_iterations++) {
    float delta = 0.0;
    for (int red_black = 0; red_black < 2; red_black++) {
      for (int y = 1; y <= HEIGHT; y++) {
        for (int x = 1; x <= WIDTH; x++) {
          if ((x%2) == ((y+red_black+1)%2)) continue;
          delta += gold_solve_gauss_seidel_loop_body(x, y, base, values, factor, divisor);
        }
      }
    }
    float average_delta = delta / N;
    if (average_delta < GOLD_SOLVE_ERROR)
      return num_iterations;
  }
  return MAX_CONVERGENCE_ITERATIONS;
}

void gold_solve_mat_mul(float *dst, float *xs, float factor, float divisor) {
  #pragma omp parallel for collapse(2)
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      dst[IDX2(idx)] = (
        + divisor*xs[IDX2(idx)]
        - factor*xs[IDX2(idx2_add(idx, idx2(-1, 0)))]
        - factor*xs[IDX2(idx2_add(idx, idx2(1, 0)))]
        - factor*xs[IDX2(idx2_add(idx, idx2(0, -1)))]
        - factor*xs[IDX2(idx2_add(idx, idx2(0, 1)))]
      );
    }
  }
}
void gold_solve_mat_sub(float *dst, float *as, float *bs) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    dst[i] = as[i] - bs[i];
  }
}
void gold_solve_mat_add(float *dst, float *as, float *bs) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    dst[i] = as[i] + bs[i];
  }
}
float gold_solve_mat_dot(float *as, float *bs) {
  float out = 0.0;
  #pragma omp parallel for reduction(+:out)
  for (int i = 0; i < N; i++) {
    out += as[i] * bs[i];
  }
  return out;
}
void gold_solve_mat_scale(float *dst, float *as, float scale) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
    dst[i] = as[i] * scale;
}
int gold_solve_conjugate_gradient(float *bs, float *xs, float factor, float divisor) {
  int num_iterations = 0;
  float *Ap = (float*)malloc(N*sizeof(float));
  float *alpha_p = (float*)malloc(N*sizeof(float));
  float *alpha_Ap = (float*)malloc(N*sizeof(float));
  float *beta_p = (float*)malloc(N*sizeof(float));
  float *Axs = (float*)malloc(N*sizeof(float));
  float *r = (float*)malloc(N*sizeof(float));
  float *p = (float*)malloc(N*sizeof(float));

  gold_solve_mat_mul(Axs, xs, factor, divisor);
  gold_solve_mat_sub(r, bs, Axs);
  for (int i = 0; i < N; i++)
    p[i] = r[i];
  float rs_old = gold_solve_mat_dot(r, r);
  for (num_iterations = 1; num_iterations <= MAX_CONVERGENCE_ITERATIONS; num_iterations++) {
    gold_solve_mat_mul(Ap, p, factor, divisor);
    float pAp = gold_solve_mat_dot(p, Ap);
    float alpha = rs_old / (pAp + GOLD_SOLVE_EPSILON);
    gold_solve_mat_scale(alpha_p, p, alpha);
    gold_solve_mat_add(xs, xs, alpha_p);
    gold_solve_mat_scale(alpha_Ap, Ap, alpha);
    gold_solve_mat_sub(r, r, alpha_Ap);
    float rs_new = gold_solve_mat_dot(r, r);
    if (rs_new < GOLD_SOLVE_ERROR) {
      goto CLEANUP;
    }
    float beta = rs_new / (rs_old + GOLD_SOLVE_EPSILON);
    gold_solve_mat_scale(beta_p, p, beta);
    gold_solve_mat_add(p, r, beta_p);
    rs_old = rs_new;
  }
CLEANUP:
  free(Ap);
  free(alpha_p);
  free(alpha_Ap);
  free(beta_p);
  free(Axs);
  free(r);
  free(p);
  return num_iterations;
}


void gold_solve_wrapper(float *d_dst, float *d_bs, float *d_xs, float factor, float divisor) {
  float *bs = (float*)malloc(N*sizeof(float));
  float *xs = (float*)malloc(N*sizeof(float));
  CUDA_CHECK(cudaMemcpy(bs, d_bs, N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(xs, d_xs, N*sizeof(float), cudaMemcpyDeviceToHost));
  gold_solve_conjugate_gradient(bs, xs, factor, divisor);
  CUDA_CHECK(cudaMemcpy(d_dst, xs, N*sizeof(float), cudaMemcpyHostToDevice));
}