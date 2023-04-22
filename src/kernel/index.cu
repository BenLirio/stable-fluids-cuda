#include <cuda_runtime.h>
#include <util/compile_options.h>
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

void kernel_step(state_t state, int current_step) {
  state_property_t *c = state.colors;
  state_property_t *x = state.x_velocities;
  state_property_t *y = state.y_velocities;
  state_property_t *p = state.pressures;
  state_property_t *d = state.divergences;

  if (USE_SOURCE_COLORS)
    kernel_source_colors<<<1, dim3(WIDTH, HEIGHT)>>>(c->previous, c->current);
  if (USE_SINK_COLORS)
    kernel_sink_colors<<<1, dim3(WIDTH, HEIGHT)>>>(c->previous, c->current);
  if (USE_DENSITY_DIFFUSE) {
    state_property_step(c);
    kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(c->previous, c->current, DIFFUSION_RATE);
  }
  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(c->previous, c->current, x->current, y->current);
  }

  if (USE_SOURCE_VELOCITIES)
    kernel_source_velocities<<<1, dim3(WIDTH, HEIGHT)>>>(x->previous, y->previous, x->current, y->current, current_step);
  if (USE_SINK_VELOCITIES)
    kernel_sink_velocities<<<1, dim3(WIDTH, HEIGHT)>>>(x->previous, y->previous, x->current, y->current);
  if (USE_VELOCITY_DIFFUSE) {
    state_property_step(x);
    kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(x->previous, x->current, VISCOSITY);
    state_property_step(y);
    kernel_diffuse<<<1, dim3(WIDTH, HEIGHT)>>>(y->previous, y->current, VISCOSITY);
    kernel_project<<<1, dim3(WIDTH, HEIGHT)>>>(x->current, y->current, p->current, d->current);
  }
  if (USE_VELOCITY_ADVECT) {
    state_property_step(x);
    state_property_step(y);
    kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(x->previous, x->current, x->previous, y->previous);
    kernel_advect<<<1, dim3(WIDTH, HEIGHT)>>>(y->previous, y->current, x->previous, y->previous);
    kernel_project<<<1, dim3(WIDTH, HEIGHT)>>>(x->current, y->current, p->current, d->current);
  }
}

void kernel_step_wrapper(state_t state, int current_step) {
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

  kernel_step(device_state, current_step);

  cudaMemcpy(state.colors->current, device_state.colors->current, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.colors->previous, device_state.colors->previous, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.x_velocities->current, device_state.x_velocities->current, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.x_velocities->previous, device_state.x_velocities->previous, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.y_velocities->current, device_state.y_velocities->current, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.y_velocities->previous, device_state.y_velocities->previous, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.pressures->current, device_state.pressures->current, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.pressures->previous, device_state.pressures->previous, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.divergences->current, device_state.divergences->current, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state.divergences->previous, device_state.divergences->previous, N*sizeof(float), cudaMemcpyDeviceToHost);

  state_cuda_free(device_state);
}