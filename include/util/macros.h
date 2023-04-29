#ifndef STABLE_FLUIDS_MACROS_H
#define STABLE_FLUIDS_MACROS_H

#include <cuda_runtime.h>
#include <stdio.h>

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

#ifndef OUTPUT
#define OUTPUT OUTPUT_PERFORMANCE
#endif

#ifndef USE_GOLD
#define USE_GOLD 0
#endif

#ifndef KERNEL_FLAGS
#define KERNEL_FLAGS 0
#endif

#define NUM_COLORS 3
#define NUM_VELOCITY_COMPONENTS 2
#define LOG_BUFFER_SIZE 4096
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}
#define NUM_NEIGHBORS 4
#define ASSERTIONS_ENABLED 0
#define VERBOSE_ASSERTIONS 0
#define EQ_THRESHOLD 0.000001f
#define MAX_AVERAGE_ERROR_THRESHOLD 0.0001f
#define MAX_SINGLE_ERROR_THRESHOLD 0.001f
#define GOLD_SOLVE_ERROR 0.000000000001f
#define GOLD_SOLVE_EPSILON 0.000000000001f
#define N (WIDTH * HEIGHT)
#define COARSENING_FACTOR 2

#define BLOCK_SIZE 32
#define BLOCK_DIM (dim3(BLOCK_SIZE, BLOCK_SIZE))
#define GRID_DIM (dim3((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE))

#define RED 0
#define BLACK 1

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

// OUTPUT FLAGS
#define OUTPUT_PERFORMANCE (1<<0)
#define OUTPUT_GIF (1<<1)
#define OUTPUT_SOLVE_ERROR (1<<2)

// KERNEL FLAGS
#define USE_SHARED_MEMORY (1<<0)
#define USE_THREAD_COARSENING (1<<1)
#define USE_ROW_COARSENING (1<<2)
#define USE_NO_BLOCK_SYNC (1<<3)
#define USE_RED_BLACK (1<<4)
#define USE_THREAD_FENCE (1<<5)
#define USE_NO_IDX (1<<6)

// TAGS
#define MILLIS_PER_SECOND 1000
#define PERFORMANCE_TAG (1<<0)
#define PERFORMANCE_TAG_STRING "[PERFORMANCE]"
#define CPU_TAG (1<<1)
#define CPU_TAG_STRING "[CPU]"
#define GPU_TAG (1<<2)
#define GPU_TAG_STRING "[GPU]"
#define TOTAL_TAG (1<<3)
#define TOTAL_TAG_STRING "[TOTAL]"
#define ADVECT_TAG (1<<4)
#define ADVECT_TAG_STRING "[ADVECT]"
#define DIFFUSE_TAG (1<<5)
#define DIFFUSE_TAG_STRING "[DIFFUSE]"
#define PROJECT_TAG (1<<6)
#define PROJECT_TAG_STRING "[PROJECT]"
#define COLOR_TAG (1<<7)
#define COLOR_TAG_STRING "[COLOR]"
#define VELOCITY_TAG (1<<8)
#define VELOCITY_TAG_STRING "[VELOCITY]"
#define SOLVE_TAG (1<<9)
#define SOLVE_TAG_STRING "[SOLVE]"

#define MAX_CONVERGENCE_ITERATIONS 400
#define CHECK_CONVERGENCE_EVERY 100
#define COLOR_SINK_RATE 0.01f
#define VELOCITY_SINK_RATE 0.01f
#define VELOCITY_SOURCE_MAGNITUDE 0.01f
#define COLOR_SOURCE_MAGNITUDE 1.0f
#define VELOCITY_SPIN_RATE 0.5f
#define MAX_COLOR 10.0f

#define USE_SOURCE_COLORS       1
#define USE_SINK_COLORS         1
#define USE_DENSITY_DIFFUSE     1
#define USE_DENSITY_ADVECT      1
#define USE_SOURCE_VELOCITIES   1
#define USE_SINK_VELOCITIES     1
#define USE_VELOCITY_DIFFUSE    1
#define USE_VELOCITY_ADVECT     1

#define RANDOMIZE_COLORS        0

#endif // STABLE_FLUIDS_MACROS_H