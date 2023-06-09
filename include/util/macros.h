#ifndef STABLE_FLUIDS_MACROS_H
#define STABLE_FLUIDS_MACROS_H

#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}
#define NUM_NEIGHBORS 4
#define ASSERTIONS_ENABLED 0
#define VERBOSE_ASSERTIONS 0
#define EQ_THRESHOLD 0.0001f
#define COLOR_SINK_RATE 0.1f
#define VELOCITY_SINK_RATE 0.1f
#define N (WIDTH * HEIGHT)

#endif // STABLE_FLUIDS_MACROS_H