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
  state_property_t *c = state.colors;
  state_property_t *x = state.x_velocities;
  state_property_t *y = state.y_velocities;
  state_property_t *p = state.pressures;
  state_property_t *d = state.divergences;
  // density
  if (USE_SOURCE_COLORS)
    gold_source_colors(c->previous, c->current);
  if (USE_SINK_COLORS)
    gold_sink_colors(c->previous, c->current);
  if (USE_DENSITY_DIFFUSE) {
    state_property_step(c);
    gold_diffuse(c->previous, c->current, DIFFUSION_RATE);
  }
  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    gold_advect(c->previous, c->current, x->current, y->current);
  }

  // velocity
  if (USE_SOURCE_VELOCITIES)
    gold_source_velocities(x->previous, y->previous, x->current, y->current, step);
  if (USE_SINK_VELOCITIES)
    gold_sink_velocities(x->previous, y->previous, x->current, y->current);
  if (USE_VELOCITY_DIFFUSE) {
    state_property_step(x);
    gold_diffuse(x->previous, x->current, VISCOSITY);
    state_property_step(y);
    gold_diffuse(y->previous, y->current, VISCOSITY);
    gold_project(x->current, y->current, p->current, d->current);
  }
  if (USE_VELOCITY_ADVECT) {
    state_property_step(x);
    state_property_step(y);
    gold_advect(x->previous, x->current, x->previous, y->previous);
    gold_advect(y->previous, y->current, x->previous, y->previous);
    gold_project(x->current, y->current, p->current, d->current);
  }
}
