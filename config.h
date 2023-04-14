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
#define ASSERTIONS_ENABLED 0
#define VERBOSE_ASSERTIONS 0
#define EQ_THRESHOLD 0.0001f
#define COLOR_SINK_RATE 0.1f
#define VELOCITY_SINK_RATE 0.1f
#define N (WIDTH * HEIGHT)


#endif // STABLE_FLUIDS_CUDA_CONFIG_H