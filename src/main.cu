#include <gold/index.h>

#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>
#include <omp.h>
#include <stdlib.h>

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
  state_t *p_state = (state_t*)malloc(sizeof(state_t));
  float *colors;

  if (USE_GOLD) {
    state_create(p_state);
    colors = p_state->all_colors[0]->cur;
  } else {
    state_cuda_create(p_state);
    if (OUTPUT&OUTPUT_GIF) colors = (float*)malloc(N*sizeof(float));
  }

  for (p_state->step = 0; p_state->step < NUM_STEPS; p_state->step++) {
    if (USE_GOLD) {
      gold_step(p_state);
    } else {
      kernel_step(p_state);
      if (OUTPUT&OUTPUT_GIF)
        CUDA_CHECK(cudaMemcpy(colors, p_state->all_colors[0]->cur, N*sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (OUTPUT&OUTPUT_GIF)
      output_gif_frame(colors, p_state->step);
  }

  if (USE_GOLD) {
    state_destroy(p_state);
  } else {
    state_cuda_destroy(p_state);
    if (OUTPUT&OUTPUT_GIF) free(colors);
  }

  return 0;
}