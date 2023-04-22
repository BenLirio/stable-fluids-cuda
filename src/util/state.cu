#include <util/state.h>
#include <util/macros.h>

void state_property_step(state_property_t *state_property_pointer) {
  float *temp = state_property_pointer->current;
  state_property_pointer->current = state_property_pointer->previous;
  state_property_pointer->previous = temp;
}

void state_property_malloc(state_property_t *property_pointer) {
  property_pointer->current = (float*)malloc(N*sizeof(float));
  property_pointer->previous = (float*)malloc(N*sizeof(float));
}
void state_property_free(state_property_t property) {
  free(property.current);
  free(property.previous);
}
void state_property_cuda_malloc(state_property_t state_property) {
  cudaMalloc(&state_property.current, N*sizeof(float));
  cudaMalloc(&state_property.previous, N*sizeof(float));
}
void state_property_cuda_free(state_property_t state_property) {
  cudaFree(state_property.current);
  cudaFree(state_property.previous);
}

void state_property_init(state_property_t property) {
  for (int i = 0; i < N; i++) {
    property.current[i] = 0;
    property.previous[i] = 0;
  }
}

void state_malloc(state_t *state_pointer) {
  state_pointer->colors = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->x_velocities = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->y_velocities = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->divergences = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->pressures = (state_property_t*)malloc(sizeof(state_property_t));
  state_property_malloc(state_pointer->colors);
  state_property_malloc(state_pointer->x_velocities);
  state_property_malloc(state_pointer->y_velocities);
  state_property_malloc(state_pointer->divergences);
  state_property_malloc(state_pointer->pressures);
}
void state_free(state_t state) {
  state_property_free(*state.colors);
  state_property_free(*state.x_velocities);
  state_property_free(*state.y_velocities);
  state_property_free(*state.divergences);
  state_property_free(*state.pressures);
  free(state.colors);
  free(state.x_velocities);
  free(state.y_velocities);
  free(state.divergences);
  free(state.pressures);
}
void state_cuda_malloc(state_t *state_pointer) {
  state_pointer->colors = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->x_velocities = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->y_velocities = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->divergences = (state_property_t*)malloc(sizeof(state_property_t));
  state_pointer->pressures = (state_property_t*)malloc(sizeof(state_property_t));
  state_property_cuda_malloc(*state_pointer->colors);
  state_property_cuda_malloc(*state_pointer->x_velocities);
  state_property_cuda_malloc(*state_pointer->y_velocities);
  state_property_cuda_malloc(*state_pointer->divergences);
  state_property_cuda_malloc(*state_pointer->pressures);
}
void state_cuda_free(state_t state) {
  state_property_cuda_free(*state.colors);
  state_property_cuda_free(*state.x_velocities);
  state_property_cuda_free(*state.y_velocities);
  state_property_cuda_free(*state.divergences);
  state_property_cuda_free(*state.pressures);
  free(state.colors);
  free(state.x_velocities);
  free(state.y_velocities);
  free(state.divergences);
  free(state.pressures);
}

void state_init(state_t state) {
  state_property_init(*state.colors);
  state_property_init(*state.x_velocities);
  state_property_init(*state.y_velocities);
  state_property_init(*state.divergences);
  state_property_init(*state.pressures);
}