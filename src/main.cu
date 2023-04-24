#include <gold/index.h>

#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>

void output_gif_frame(float *colors, int i) {
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
    if (OUTPUT&OUTPUT_GIF) {
      cudaMemcpy(colors, device_state.colors->current, N*sizeof(float), cudaMemcpyDeviceToHost);
      output_gif_frame(colors, step);
    }
  }
  free(colors);

  state_cuda_free(device_state);
}

void run_with_gold(state_t state) {
  for (int step = 0; step < NUM_STEPS; step++) {
    gold_step(state, step);
    if (OUTPUT&OUTPUT_GIF)
      output_gif_frame(state.colors->current, step);
  }
}

int main() {
  state_t state;
  state_malloc(&state);
  state_init(state);

  if (USE_GOLD) {
    run_with_gold(state);
  } else {
    run_with_kernel(state);
  }

  state_free(state);
  return 0;
}
