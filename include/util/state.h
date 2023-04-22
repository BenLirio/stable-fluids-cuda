#ifndef STABLE_FLUIDS_CUDA_STATE_H
#define STABLE_FLUIDS_CUDA_STATE_H

struct state_property {
  float *previous;
  float *current;
};
typedef struct state_property state_property_t;

void state_property_malloc(state_property_t *property);
void state_property_free(state_property_t property);
void state_property_init(state_property_t property);
void state_property_step(state_property_t *property);

void state_property_cuda_malloc(state_property_t property);
void state_property_cuda_free(state_property_t property);

struct state {
  state_property_t *colors;
  state_property_t *x_velocities;
  state_property_t *y_velocities;
  state_property_t *pressures;
  state_property_t *divergences;
};
typedef struct state state_t;

void state_malloc(state_t *state);
void state_free(state_t state);
void state_init(state_t state);

void state_cuda_malloc(state_t *state);
void state_cuda_free(state_t state);

#endif // STABLE_FLUIDS_CUDA_STATE_H