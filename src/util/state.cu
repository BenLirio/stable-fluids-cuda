#include <util/state.h>
#include <util/macros.h>
#include <stdio.h>

void state_property_step(state_property_t *state_property_pointer) {
  float *temp = state_property_pointer->cur;
  state_property_pointer->cur = state_property_pointer->prev;
  state_property_pointer->prev = temp;
}

void _state_create(state_t *state) {
  state->step = 0;
  state->log_buffer_filled = 0;
  state->log_buffer_index = 0;
  CUDA_CHECK(cudaEventCreate(&state->start));
  CUDA_CHECK(cudaEventRecord(state->start));
  for (int i = 0; i < NUM_COLORS; i++) {
    state->all_colors[i] = (state_property_t*)malloc(sizeof(state_property_t));
  }
  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    state->all_velocities[i] = (state_property_t*)malloc(sizeof(state_property_t));
  }
}

void state_create(state_t *state) {

  _state_create(state);

  for (int i = 0; i < NUM_COLORS; i++) {
    state->all_colors[i]->cur = (float*)malloc(N*sizeof(float));
    state->all_colors[i]->prev = (float*)malloc(N*sizeof(float));
    for (int j = 0; j < N; j++) {
      state->all_colors[i]->cur[j] = 0.0;
      state->all_colors[i]->prev[j] = 0.0;
    }
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    state->all_velocities[i]->cur = (float*)malloc(N*sizeof(float));
    state->all_velocities[i]->prev = (float*)malloc(N*sizeof(float));
    for (int j = 0; j < N; j++) {
      state->all_velocities[i]->cur[j] = 0.0;
      state->all_velocities[i]->prev[j] = 0.0;
    }

  }
}

void state_cuda_create(state_t *state) {
  _state_create(state);

  for (int i = 0; i < NUM_COLORS; i++) {
    CUDA_CHECK(cudaMalloc(&state->all_colors[i]->cur, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->all_colors[i]->prev, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(state->all_colors[i]->cur, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(state->all_colors[i]->prev, 0, N*sizeof(float)));
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    CUDA_CHECK(cudaMalloc(&state->all_velocities[i]->cur, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->all_velocities[i]->prev, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(state->all_velocities[i]->cur, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(state->all_velocities[i]->prev, 0, N*sizeof(float)));
  }
}

void state_destroy(state_t *state) {

  cudaEventDestroy(state->start);

  for (int i = 0; i < NUM_COLORS; i++) {
    free(state->all_colors[i]->cur);
    free(state->all_colors[i]->prev);
    free(state->all_colors[i]);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    free(state->all_velocities[i]->cur);
    free(state->all_velocities[i]->prev);
    free(state->all_velocities[i]);
  }

  free(state);
}

void state_cuda_destroy(state_t *state) {

  cudaEventDestroy(state->start);

  for (int i = 0; i < NUM_COLORS; i++) {
    cudaFree(state->all_colors[i]->cur);
    cudaFree(state->all_colors[i]->prev);
    free(state->all_colors[i]);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    cudaFree(state->all_velocities[i]->cur);
    cudaFree(state->all_velocities[i]->prev);
    free(state->all_velocities[i]);
  }

  free(state);
}