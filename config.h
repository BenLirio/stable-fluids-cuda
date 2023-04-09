#ifndef STABLE_FLUIDS_CUDA_CONFIG_H
#define STABLE_FLUIDS_CUDA_CONFIG_H

#ifndef NUM_STEPS
#define NUM_STEPS 120
#endif

#ifndef WIDTH
#define WIDTH 64
#endif

#ifndef HEIGHT
#define HEIGHT 64
#endif

#ifndef DIFFUSION_RATE
#define DIFFUSION_RATE 0.01f
#endif

#ifndef VISCOSITY
#define VISCOSITY 0.0001f
#endif

#ifndef GAUSS_SEIDEL_ITERATIONS
#define GAUSS_SEIDEL_ITERATIONS 20
#endif

#ifndef TIME_STEP
#define TIME_STEP 0.01f
#endif



#define NUM_NEIGHBORS 4
#define IDX2(idx2) ((idx2.y-1) * WIDTH + (idx2.x-1))
#define IDX2_0_OFFSET(idx2) (((idx2.y) * (WIDTH)) + (idx2.x))
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}
#define N (WIDTH * HEIGHT)
#define ASSERTIONS_ENABLED 1
#define VERBOSE_ASSERTIONS 1


#endif // STABLE_FLUIDS_CUDA_CONFIG_H