#include <cuda_runtime.h>
#include <kernel/diffuse.cuh>
#include <kernel/advect.cuh>
#include <kernel/project.cuh>
#include <util/macros.h>
#include <kernel/index.cuh>
#include <kernel/source_colors.cuh>
#include <kernel/source_velocities.cuh>
#include <kernel/sink_velocities.cuh>
#include <kernel/sink_colors.cuh>
#include <util/state.h>
#include <stdio.h>
#include <util/performance.cuh>

void kernel_color_step(state_t *p_state) {
  performance_t *performance_ptr;
  performance_malloc(&performance_ptr);

  int step = p_state->step;
  state_property_t *c = p_state->all_colors[0];
  state_property_t *x = p_state->all_velocities[0];
  state_property_t *y = p_state->all_velocities[1];

  if (USE_SOURCE_COLORS) kernel_source_colors<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_SINK_COLORS) kernel_sink_colors<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_DENSITY_DIFFUSE) {
    state_property_step(c);
    performance_start(performance_ptr);
    kernel_diffuse_wrapper(step, c->prev, c->cur, DIFFUSION_RATE);
    performance_record(performance_ptr, step, DIFFUSE_TAG|COLOR_TAG);
  }

  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    performance_start(performance_ptr);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur, x->cur, y->cur);
    CUDA_CHECK(cudaPeekAtLastError());
    performance_record(performance_ptr, step, ADVECT_TAG|COLOR_TAG);
  }

  performance_free(performance_ptr);
}

void kernel_velocity_step(state_t *p_state) {
  performance_t *performance_ptr;
  performance_malloc(&performance_ptr);

  int step = p_state->step;
  state_property_t *x = p_state->all_velocities[0];
  state_property_t *y = p_state->all_velocities[1];
  float *p, *d;
  CUDA_CHECK(cudaMalloc(&p, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d, N*sizeof(float)));


  if (USE_SOURCE_VELOCITIES) kernel_source_velocities<<<GRID_DIM, BLOCK_DIM>>>(x->prev, y->prev, x->cur, y->cur, step);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_SINK_VELOCITIES) kernel_sink_velocities<<<GRID_DIM, BLOCK_DIM>>>(x->prev, y->prev, x->cur, y->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_VELOCITY_DIFFUSE) {
    state_property_step(x);
    performance_start(performance_ptr);
    kernel_diffuse_wrapper(step, x->prev, x->cur, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    state_property_step(y);
    performance_start(performance_ptr);
    kernel_diffuse_wrapper(step, y->prev, y->cur, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    kernel_project_wrapper(step, x->cur, y->cur, p, d);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }

  if (USE_VELOCITY_ADVECT) {
    state_property_step(x);
    state_property_step(y);

    performance_start(performance_ptr);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(x->prev, x->cur, x->prev, y->prev);
    CUDA_CHECK(cudaPeekAtLastError());
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(y->prev, y->cur, x->prev, y->prev);
    CUDA_CHECK(cudaPeekAtLastError());
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    kernel_project_wrapper(step, x->cur, y->cur, p, d);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }

  cudaFree(p);
  cudaFree(d);
  performance_free(performance_ptr);
}

void kernel_step(state_t *p_state) {
  kernel_color_step(p_state);
  kernel_velocity_step(p_state);
}
