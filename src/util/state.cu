#include <util/state.h>
#include <util/macros.h>
#include <stdio.h>

void state_property_step(state_property_t *state_property_pointer) {
  float *temp = state_property_pointer->cur;
  state_property_pointer->cur = state_property_pointer->prev;
  state_property_pointer->prev = temp;
}

void state_create(state_t *p_state) {

  p_state->step = 0;

  CUDA_CHECK(cudaEventCreate(&p_state->start));

  for (int i = 0; i < NUM_COLORS; i++) {
    p_state->all_colors[i] = (state_property_t*)malloc(sizeof(state_property_t));
    p_state->all_colors[i]->cur = (float*)malloc(N*sizeof(float));
    p_state->all_colors[i]->prev = (float*)malloc(N*sizeof(float));
    for (int j = 0; j < N; j++) {
      p_state->all_colors[i]->cur[j] = 0.0;
      p_state->all_colors[i]->prev[j] = 0.0;
    }
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    p_state->all_velocities[i] = (state_property_t*)malloc(sizeof(state_property_t));
    p_state->all_velocities[i]->cur = (float*)malloc(N*sizeof(float));
    p_state->all_velocities[i]->prev = (float*)malloc(N*sizeof(float));
    for (int j = 0; j < N; j++) {
      p_state->all_velocities[i]->cur[j] = 0.0;
      p_state->all_velocities[i]->prev[j] = 0.0;
    }

  }
}

void state_cuda_create(state_t *p_state) {

  p_state->step = 0;

  CUDA_CHECK(cudaEventCreate(&p_state->start));

  for (int i = 0; i < NUM_COLORS; i++) {
    p_state->all_colors[i] = (state_property_t*)malloc(sizeof(state_property_t));
    CUDA_CHECK(cudaMalloc(&p_state->all_colors[i]->cur, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_state->all_colors[i]->prev, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(p_state->all_colors[i]->cur, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(p_state->all_colors[i]->prev, 0, N*sizeof(float)));
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    p_state->all_velocities[i] = (state_property_t*)malloc(sizeof(state_property_t));
    CUDA_CHECK(cudaMalloc(&p_state->all_velocities[i]->cur, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_state->all_velocities[i]->prev, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(p_state->all_velocities[i]->cur, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(p_state->all_velocities[i]->prev, 0, N*sizeof(float)));
  }
}

void state_destroy(state_t *p_state) {

  cudaEventDestroy(p_state->start);

  for (int i = 0; i < NUM_COLORS; i++) {
    free(p_state->all_colors[i]->cur);
    free(p_state->all_colors[i]->prev);
    free(p_state->all_colors[i]);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    free(p_state->all_velocities[i]->cur);
    free(p_state->all_velocities[i]->prev);
    free(p_state->all_velocities[i]);
  }

  free(p_state);
}

void state_cuda_destroy(state_t *p_state) {

  cudaEventDestroy(p_state->start);

  for (int i = 0; i < NUM_COLORS; i++) {
    cudaFree(p_state->all_colors[i]->cur);
    cudaFree(p_state->all_colors[i]->prev);
    free(p_state->all_colors[i]);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    cudaFree(p_state->all_velocities[i]->cur);
    cudaFree(p_state->all_velocities[i]->prev);
    free(p_state->all_velocities[i]);
  }

  free(p_state);
}