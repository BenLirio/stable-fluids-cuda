#include <gold/index.h>
#include <util/macros.h>

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
#include <util/performance.cuh>

void gold_step(state_t state, int step) {
  performance_t *performance_ptr;  
  performance_malloc(&performance_ptr);

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
    performance_start(performance_ptr);
    gold_diffuse(c->previous, c->current, DIFFUSION_RATE);
    performance_record(performance_ptr, step, DIFFUSE_TAG|COLOR_TAG);
  }

  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    performance_start(performance_ptr);
    gold_advect(c->previous, c->current, x->current, y->current);
    performance_record(performance_ptr, step, ADVECT_TAG|COLOR_TAG);
  }

  // velocity
  if (USE_SOURCE_VELOCITIES)
    gold_source_velocities(x->previous, y->previous, x->current, y->current, step);

  if (USE_SINK_VELOCITIES)
    gold_sink_velocities(x->previous, y->previous, x->current, y->current);

  if (USE_VELOCITY_DIFFUSE) {

    state_property_step(x);
    performance_start(performance_ptr);
    gold_diffuse(x->previous, x->current, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    state_property_step(y);
    performance_start(performance_ptr);
    gold_diffuse(y->previous, y->current, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_project(x->current, y->current, p->current, d->current);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }
  if (USE_VELOCITY_ADVECT) {

    state_property_step(x);
    state_property_step(y);
    performance_start(performance_ptr);
    gold_advect(x->previous, x->current, x->previous, y->previous);
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_advect(y->previous, y->current, x->previous, y->previous);
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_project(x->current, y->current, p->current, d->current);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }
  performance_free(performance_ptr);
}
