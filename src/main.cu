#include <gold/index.h>
#include <util/compile_options.h>
#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

void output_color(float *colors, int i) {
  if (i != 0)
    printf(",");
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      printf("%f", colors[y*WIDTH+x]);
      if (y != HEIGHT - 1 || x != WIDTH - 1)
        printf(",");
    }
  }
}

void run_with_kernel(state_t state) {
  state_t device_state;
  state_cuda_malloc(&device_state);

  cudaMemcpy(device_state.colors->current, state.colors->current, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.colors->previous, state.colors->previous, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.x_velocities->current, state.x_velocities->current, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.x_velocities->previous, state.x_velocities->previous, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.y_velocities->current, state.y_velocities->current, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.y_velocities->previous, state.y_velocities->previous, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.pressures->current, state.pressures->current, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.pressures->previous, state.pressures->previous, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.divergences->current, state.divergences->current, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_state.divergences->previous, state.divergences->previous, N*sizeof(float), cudaMemcpyHostToDevice);


  float *colors = (float*)malloc(N*sizeof(float));
  for (int step = 0; step < NUM_STEPS; step++) {
    kernel_step(device_state, step);
    cudaMemcpy(colors, device_state.colors->current, N*sizeof(float), cudaMemcpyDeviceToHost);
    output_color(colors, step);
  }
  free(colors);

  state_cuda_free(device_state);
}

int main() {
  int current_step = 0;
  state_t state;
  state_malloc(&state);
  state_init(state);

  run_with_kernel(state);

  // void (*step)(state_t, int) = USE_GOLD ? gold_step : kernel_step_wrapper;
  // for (int i = 0; i < NUM_STEPS; i++) {
  //   output_color(state.colors->current, current_step);
  //   gold_step(state, current_step);
  //   current_step++;
  // }

  state_free(state);
  return 0;
}
