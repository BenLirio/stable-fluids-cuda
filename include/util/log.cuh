#ifndef STABLE_FLUIDS_CUDA_LOG_H_
#define STABLE_FLUIDS_CUDA_LOG_H_
#include <util/state.h>

int log_with_error_and_gauss_step(state_t *state, int id, int tags, float error, int guass_step);
int log_with_error(state_t *state, int id, int tags, float error);
int log(state_t *state, int id, int tags);

void empty_log_buffer(state_t *state);

#endif // STABLE_FLUIDS_CUDA_LOG_H_