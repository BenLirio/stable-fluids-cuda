#ifndef STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_
#define STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_

int gold_solve_gauss_seidel(float *bs, float *xs, float factor, float divisor);
int gold_solve_conjugate_gradient(float *bs, float *xs, float factor, float divisor);
void gold_solve_wrapper(float *d_dst, float *d_bs, float *d_xs, float factor, float divisor);

#endif // STABLE_FLUIDS_CUDA_GOLD_SOLVE_H_