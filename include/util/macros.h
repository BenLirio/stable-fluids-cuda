#ifndef STABLE_FLUIDS_MACROS_H
#include <cuda_runtime.h>
#define STABLE_FLUIDS_MACROS_H

#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}
#define NUM_NEIGHBORS 4
#define ASSERTIONS_ENABLED 0
#define VERBOSE_ASSERTIONS 0
#define EQ_THRESHOLD 0.0001f
#define MAX_AVERAGE_ERROR_THRESHOLD 0.0001f
#define MAX_SINGLE_ERROR_THRESHOLD 0.001f
#define N (WIDTH * HEIGHT)

#define BLOCK_SIZE 32
#define BLOCK_DIM (dim3(BLOCK_SIZE, BLOCK_SIZE))
#define GRID_DIM (dim3((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE))

#define RED 0
#define BLACK 1

#define MILLIS_PER_SECOND 1000
#define OUTPUT_PERFORMANCE 0
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

#define MAX_CONVERGENCE_ITERATIONS 10000
#define CHECK_CONVERGENCE_EVERY 100
#define COLOR_SINK_RATE 0.1f
#define VELOCITY_SINK_RATE 0.01f
#define VELOCITY_SOURCE_MAGNITUDE 10.0f
#define COLOR_SOURCE_MAGNITUDE 1.0f
#define VELOCITY_SPIN_RATE 10.0f

#define USE_GOLD                0

#define USE_SOURCE_COLORS       1
#define USE_SINK_COLORS         1
#define USE_DENSITY_DIFFUSE     1
#define USE_DENSITY_ADVECT      1
#define USE_SOURCE_VELOCITIES   1
#define USE_SINK_VELOCITIES     1
#define USE_VELOCITY_DIFFUSE    1
#define USE_VELOCITY_ADVECT     1

#define RANDOMIZE_COLORS        1

#endif // STABLE_FLUIDS_MACROS_H