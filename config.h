#ifndef STABLE_FLUIDS_CUDA_CONFIG_H
#define STABLE_FLUIDS_CUDA_CONFIG_H

#define NUM_STEPS 100
#define WIDTH 64
#define HEIGHT 64
#define NUM_NEIGHBORS 4
#define DIFFUSION_RATE 0.01f
#define TIME_STEP 0.01f
#define GAUSS_SEIDEL_ITERATIONS 20
#define IDX(y, x) ((y) * WIDTH + (x))
#define DIDX(dim) ((dim.y) * WIDTH + (dim.x))
#define IDX2(idx2) ((idx2.y) * WIDTH + (idx2.x))
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

#define N (WIDTH * HEIGHT)


#endif // STABLE_FLUIDS_CUDA_CONFIG_H