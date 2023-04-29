#include <gold/index.h>

#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>
#include <omp.h>
#include <stdlib.h>
#include <util/log.cuh>

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

int main() {
  state_t *state = (state_t*)malloc(sizeof(state_t));
  float *colors;

  if (USE_GOLD) {
    state_create(state);
    colors = state->all_colors[0]->cur;
  } else {
    state_cuda_create(state);
    if (OUTPUT&OUTPUT_GIF) colors = (float*)malloc(N*sizeof(float));
  }

  for (state->step = 0; state->step < NUM_STEPS; state->step++) {
    empty_log_buffer(state);
    if (USE_GOLD) {
      gold_step(state);
    } else {
      kernel_step(state);
      if (OUTPUT&OUTPUT_GIF)
        CUDA_CHECK(cudaMemcpy(colors, state->all_colors[0]->cur, N*sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (OUTPUT&OUTPUT_GIF)
      output_gif_frame(colors, state->step);
  }

  if (USE_GOLD) {
    state_destroy(state);
  } else {
    state_cuda_destroy(state);
    if (OUTPUT&OUTPUT_GIF) free(colors);
  }

  return 0;
}