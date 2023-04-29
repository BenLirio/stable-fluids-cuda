#ifndef STABLE_FLUIDS_CUDA_STATE_H
#define STABLE_FLUIDS_CUDA_STATE_H
#include <util/macros.h>

struct state_property {
  float *prev;
  float *cur;
};
typedef struct state_property state_property_t;
void state_property_step(state_property_t *property);

struct state {
  int step;
  cudaEvent_t start;
  state_property_t *all_colors[NUM_COLORS];
  state_property_t *all_velocities[NUM_VELOCITY_COMPONENTS];
};
typedef struct state state_t;


void state_create(state_t *p_state);
void state_destroy(state_t *state);

void state_cuda_create(state_t *p_state);
void state_cuda_destroy(state_t *state);


#endif // STABLE_FLUIDS_CUDA_STATE_H