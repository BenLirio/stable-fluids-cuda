#include <util/state.h>
#include <util/macros.h>
#include <stdio.h>
#include <gif_lib.h>

int _write_to_stdout(GifFileType *gif_file, const GifByteType *data, int length) {
  return fwrite(data, 1, length, stdout);
}

void state_property_step(state_property_t *state_property_pointer) {
  float *temp = state_property_pointer->cur;
  state_property_pointer->cur = state_property_pointer->prev;
  state_property_pointer->prev = temp;
}

void _state_create(state_t *state) {
  if (OUTPUT&OUTPUT_GIF) {
    int error;
    state->gif_dst = EGifOpen(NULL, _write_to_stdout, &error);
    if (!state->gif_dst) {
      fprintf(stderr, "EGifOpenFileName() failed - %d\n", error);
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_SHADES; i++) {
      state->gif_palette[i].Red = i;
      state->gif_palette[i].Green = i;
      state->gif_palette[i].Blue = i;
    }
    state->gif_dst->SWidth = OUTPUT_WIDTH;
    state->gif_dst->SHeight = OUTPUT_HEIGHT;
    state->gif_dst->SColorResolution = 8;
    state->gif_dst->SBackGroundColor = 0;
    state->gif_dst->SColorMap = GifMakeMapObject(NUM_SHADES, state->gif_palette);
  }

  state->step = 0;
  state->log_buffer_filled = 0;
  state->log_buffer_index = 0;
  state->depth = 0;
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

void _state_destroy(state_t *state) {
  cudaEventDestroy(state->start);
  for (int i = 0; i < NUM_COLORS; i++) {
    free(state->all_colors[i]);
  }
  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    free(state->all_velocities[i]);
  }
  free(state);
}

void _state_write_gif(state_t *state) {
  if (!(OUTPUT&OUTPUT_GIF)) return;
  int error;
  if (EGifSpew(state->gif_dst) == GIF_ERROR) {
    fprintf(stderr, "EGifSpew() failed - %d\n", state->gif_dst->Error);
    EGifCloseFile(state->gif_dst, &error);
    exit(EXIT_FAILURE);
  }
}

void state_destroy(state_t *state) {
  _state_write_gif(state);

  for (int i = 0; i < NUM_COLORS; i++) {
    free(state->all_colors[i]->cur);
    free(state->all_colors[i]->prev);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    free(state->all_velocities[i]->cur);
    free(state->all_velocities[i]->prev);
  }
  _state_destroy(state);
}

void state_cuda_destroy(state_t *state) {
  _state_write_gif(state);

  for (int i = 0; i < NUM_COLORS; i++) {
    cudaFree(state->all_colors[i]->cur);
    cudaFree(state->all_colors[i]->prev);
  }

  for (int i = 0; i < NUM_VELOCITY_COMPONENTS; i++) {
    cudaFree(state->all_velocities[i]->cur);
    cudaFree(state->all_velocities[i]->prev);
  }

  _state_destroy(state);
}

void state_push(state_t *state) {
  state->depth++;
}

void state_pop(state_t *state) {
  state->depth--;
}