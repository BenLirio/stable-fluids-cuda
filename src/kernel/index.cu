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
#include <util/log.cuh>

void kernel_color_step(state_t *state) {

  state_property_t *c = state->all_colors[0];
  state_property_t *x = state->all_velocities[0];
  state_property_t *y = state->all_velocities[1];

  if (USE_SOURCE_COLORS) kernel_source_colors<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_SINK_COLORS) kernel_sink_colors<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_DENSITY_DIFFUSE) {
    state_property_step(c);
    int id = log(state, rand(), DIFFUSE_TAG|COLOR_TAG);
    kernel_diffuse_wrapper(state, c->prev, c->cur, DIFFUSION_RATE);
    log(state, id, DIFFUSE_TAG|COLOR_TAG);
  }

  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    int id = log(state, rand(), ADVECT_TAG|COLOR_TAG);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(c->prev, c->cur, x->cur, y->cur);
    CUDA_CHECK(cudaPeekAtLastError());
    log(state, id, ADVECT_TAG|COLOR_TAG);
  }

}

void kernel_velocity_step(state_t *state) {

  int id;
  int step = state->step;
  state_property_t *x = state->all_velocities[0];
  state_property_t *y = state->all_velocities[1];
  float *p, *d;
  CUDA_CHECK(cudaMalloc(&p, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d, N*sizeof(float)));


  if (USE_SOURCE_VELOCITIES) kernel_source_velocities<<<GRID_DIM, BLOCK_DIM>>>(x->prev, y->prev, x->cur, y->cur, step);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_SINK_VELOCITIES) kernel_sink_velocities<<<GRID_DIM, BLOCK_DIM>>>(x->prev, y->prev, x->cur, y->cur);
  CUDA_CHECK(cudaPeekAtLastError());

  if (USE_VELOCITY_DIFFUSE) {
    state_property_step(x);
    id = log(state, rand(), DIFFUSE_TAG|VELOCITY_TAG);
    kernel_diffuse_wrapper(state, x->prev, x->cur, VISCOSITY);
    log(state, id, DIFFUSE_TAG|VELOCITY_TAG);

    state_property_step(y);
    id = log(state, rand(), DIFFUSE_TAG|VELOCITY_TAG);
    kernel_diffuse_wrapper(state, y->prev, y->cur, VISCOSITY);
    log(state, id, DIFFUSE_TAG|VELOCITY_TAG);

    id = log(state, rand(), PROJECT_TAG);
    kernel_project_wrapper(state, x->cur, y->cur, p, d);
    log(state, id, PROJECT_TAG);
  }

  if (USE_VELOCITY_ADVECT) {
    state_property_step(x);
    state_property_step(y);

    id = log(state, rand(), ADVECT_TAG|VELOCITY_TAG);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(x->prev, x->cur, x->prev, y->prev);
    CUDA_CHECK(cudaPeekAtLastError());
    log(state, id, ADVECT_TAG|VELOCITY_TAG);

    id = log(state, rand(), ADVECT_TAG|VELOCITY_TAG);
    kernel_advect<<<GRID_DIM, BLOCK_DIM>>>(y->prev, y->cur, x->prev, y->prev);
    CUDA_CHECK(cudaPeekAtLastError());
    log(state, id, ADVECT_TAG|VELOCITY_TAG);

    id = log(state, rand(), PROJECT_TAG);
    kernel_project_wrapper(state, x->cur, y->cur, p, d);
    log(state, id, PROJECT_TAG);
  }

  cudaFree(p);
  cudaFree(d);
}

void kernel_step(state_t *state) {
  kernel_color_step(state);
  kernel_velocity_step(state);
}
