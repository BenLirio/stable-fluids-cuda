#include <util/performance.cuh>
#include <cuda_runtime.h>
#include <util/macros.h>
#include <stdio.h>

void performance_malloc(performance_t **performance_ptr_ptr) {
  if (!OUTPUT_PERFORMANCE) return;
  *performance_ptr_ptr = (performance_t *)malloc(sizeof(performance_t));
  performance_t *performance_ptr = *performance_ptr_ptr;
  cudaEventCreate(&performance_ptr->cuda_event_start);
  cudaEventCreate(&performance_ptr->cuda_event_stop);
}
void performance_free(performance_t *performance_ptr) {
  if (!OUTPUT_PERFORMANCE) return;
  cudaEventDestroy(performance_ptr->cuda_event_start);
  cudaEventDestroy(performance_ptr->cuda_event_stop);
  free(performance_ptr);
}

void performance_start_clock(performance_t *performance_ptr) {
  if (!OUTPUT_PERFORMANCE) return;
  performance_ptr->clock_start = clock();
}

void print_tags(int tags) {
  if (tags&PERFORMANCE_TAG) {
    printf(PERFORMANCE_TAG_STRING);
  }
  if (tags&CPU_TAG) {
    printf(CPU_TAG_STRING);
  }
  if (tags&GPU_TAG) {
    printf(GPU_TAG_STRING);
  }
  if (tags&TOTAL_TAG) {
    printf(TOTAL_TAG_STRING);
  }
  if (tags&ADVECT_TAG) {
    printf(ADVECT_TAG_STRING);
  }
  if (tags&DIFFUSE_TAG) {
    printf(DIFFUSE_TAG_STRING);
  }
  if (tags&PROJECT_TAG) {
    printf(PROJECT_TAG_STRING);
  }
  if (tags&COLOR_TAG) {
    printf(COLOR_TAG_STRING);
  }
  if (tags&VELOCITY_TAG) {
    printf(VELOCITY_TAG_STRING);
  }
}
void print_step(int step) {
  printf("[step=%d]", step);
}

void performance_record_clock(performance_t *performance_ptr, int step, int tags) {
  if (!OUTPUT_PERFORMANCE) return;
  clock_t delta = clock() - performance_ptr->clock_start;
  double elapsed_time = (double)delta / CLOCKS_PER_SEC*MILLIS_PER_SECOND;
  print_tags(tags);
  print_step(step);
  printf("[time=%lf]", elapsed_time);
  printf("\n");
}
void performance_start_cuda_event(performance_t *performance_ptr) {
  if (!OUTPUT_PERFORMANCE) return;
  cudaEventRecord(performance_ptr->cuda_event_start);
}

void performance_record_cuda_event(performance_t *performance_ptr, int step, int tags) {
  if (!OUTPUT_PERFORMANCE) return;
  cudaEventRecord(performance_ptr->cuda_event_stop);
  cudaEventSynchronize(performance_ptr->cuda_event_stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, performance_ptr->cuda_event_start, performance_ptr->cuda_event_stop);
  print_tags(tags);
  print_step(step);
  printf("[time=%f]", elapsed_time);
  printf("\n");
}