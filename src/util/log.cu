#include <util/macros.h>
#include <stdio.h>
#include <util/state.h>
#include <cuda_runtime.h>
#include <util/performance.cuh>

int log(state_t *state, int id, int tags, float error, int guass_step) {
  if (!(OUTPUT&OUTPUT_PERFORMANCE)) return 0;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ellapsed_time;
  CUDA_CHECK(cudaEventElapsedTime(&ellapsed_time, state->start, stop));

  state->log_buffer[(state->log_buffer_index + state->log_buffer_filled++)%LOG_BUFFER_SIZE] = (log_entry_t) {
    .id = id,
    .step = state->step,
    .time = ellapsed_time,
    .tags = tags,
    .error = error,
    .guass_step = guass_step
  };
  return id;
}

int log (state_t *state, int id, int tags, float error) {
  return log(state, id, tags, error, 0);
}

int log(state_t *state, int id, int tags) {
  return log(state, id, tags, 0.0f);
}

void empty_log_buffer(state_t *state) {
  if (!(OUTPUT&OUTPUT_PERFORMANCE)) return;
  if (state->log_buffer_filled > LOG_BUFFER_SIZE) {
    printf("Log buffer overflow\n");
    exit(1);
  }
  for(;state->log_buffer_filled > 0; state->log_buffer_filled--) {
    int idx = state->log_buffer_index++ % LOG_BUFFER_SIZE;
    log_entry_t entry = state->log_buffer[idx];
    print_tags(entry.tags);
    printf("[time=%f][step=%d][id=%d][error=%f][gauss_step=%d]\n", entry.time, entry.step, entry.id, entry.error, entry.guass_step);
  }
}