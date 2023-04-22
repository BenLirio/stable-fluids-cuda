#include <gold/index.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <gold/advect.h>
#include <gold/diffuse.h>
#include <gold/project.h>
#include <util/vec2.cuh>
#include <util/idx2.cuh>
#include <util/state.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <kernel/diffuse.cuh>

#include <gold/sink_colors.cuh>
#include <gold/source_colors.cuh>
#include <gold/source_velocities.cuh>
#include <gold/sink_velocities.cuh>

void gold_step(state_t state, int step) {
  float *c = state.colors->current;
  float *c0 = state.colors->previous;

  float *x = state.x_velocities->current;
  float *x0 = state.x_velocities->previous;

  float *y = state.y_velocities->current;
  float *y0 = state.y_velocities->previous;

  float *p = state.pressures->current;
  float *d = state.divergences->current;
  // density
  gold_source_colors(c0, c);
  gold_sink_colors(c0, c);
  state_property_step(state.colors);
  c = state.colors->current;
  c0 = state.colors->previous;
  gold_diffuse(c0, c, DIFFUSION_RATE);
  gold_advect(c0, c, x, y);

  // velocity
  gold_source_velocities(x0, y0, x, y, step);
  gold_sink_velocities(x0, y0, x, y);
  state_property_step(state.x_velocities);
  x = state.x_velocities->current;
  x0 = state.x_velocities->previous;
  gold_diffuse(x0, x, VISCOSITY);
  state_property_step(state.y_velocities);
  y = state.y_velocities->current;
  y0 = state.y_velocities->previous;
  gold_diffuse(y0, y, VISCOSITY);
  gold_project(x, y, p, d);

  state_property_step(state.x_velocities);
  x = state.x_velocities->current;
  x0 = state.x_velocities->previous;
  state_property_step(state.y_velocities);
  y = state.y_velocities->current;
  y0 = state.y_velocities->previous;
  gold_advect(x0, x, x0, y0);
  gold_advect(y0, y, x0, y0);
  gold_project(x, y, p, d);
}
