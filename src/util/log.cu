
int log_time_start(state_t *state) {
  cudaEvent_t t;
  cudaEventCreate(&t);
  cudaEventRecord(t);
  cudaEventDestroy(t);
}