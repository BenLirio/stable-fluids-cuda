#ifndef STABLE_FLUIDS_CUDA_PERFORMANCE_H_
#define STABLE_FLUIDS_CUDA_PERFORMANCE_H_

struct performance_t {
  clock_t clock_start;
  cudaEvent_t cuda_event_start;
  cudaEvent_t cuda_event_stop;
};
typedef struct performance_t performance_t;


void performance_free(performance_t *performance_ptr);
void performance_malloc(performance_t **performance_ptr_ptr);

void performance_start_clock(performance_t *performance_ptr);
void performance_record_clock(performance_t *performance_ptr, int step, int tags);
void performance_start_cuda_event(performance_t *performance_ptr);
void performance_record_cuda_event(performance_t *performance_ptr, int step, int tags);

#endif // STABLE_FLUIDS_CUDA_PERFORMANCE_H_