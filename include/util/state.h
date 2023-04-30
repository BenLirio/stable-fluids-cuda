#ifndef STABLE_FLUIDS_CUDA_STATE_H
#define STABLE_FLUIDS_CUDA_STATE_H
#include <util/macros.h>
#include <gif_lib.h>

// defined here to avoid cyclic dependency
struct log_entry {
  int id;
  int step;
  float time;
  int tags;
  float error;
  int guass_step;
  int depth;
};
typedef struct log_entry log_entry_t;



struct state_property {
  float *prev;
  float *cur;
};
typedef struct state_property state_property_t;
void state_property_step(state_property_t *property);

struct state {
  int step;
  cudaEvent_t start;
  log_entry_t log_buffer[LOG_BUFFER_SIZE];
  int log_buffer_index;
  int log_buffer_filled;
  int depth;
  GifFileType *gif_dst;
  GifColorType gif_palette[NUM_SHADES];
  state_property_t *all_colors[NUM_COLORS];
  state_property_t *all_velocities[NUM_VELOCITY_COMPONENTS];
};
typedef struct state state_t;


void state_create(state_t *state);
void state_destroy(state_t *state);

void state_cuda_create(state_t *state);
void state_cuda_destroy(state_t *state);

void state_push(state_t *state);
void state_pop(state_t *state);


#endif // STABLE_FLUIDS_CUDA_STATE_H