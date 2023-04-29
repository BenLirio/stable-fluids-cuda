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

void gold_step(state *p_state) {
  performance_t *performance_ptr;  
  performance_malloc(&performance_ptr);

  int step = p_state->step;
  state_property_t *c = p_state->all_colors[0];
  state_property_t *x = p_state->all_velocities[0];
  state_property_t *y = p_state->all_velocities[1];
  float *p = (float*)malloc(N*sizeof(float));
  float *d = (float*)malloc(N*sizeof(float));

  // density
  if (USE_SOURCE_COLORS)
    gold_source_colors(c->prev, c->cur);

  if (USE_SINK_COLORS)
    gold_sink_colors(c->prev, c->cur);

  if (USE_DENSITY_DIFFUSE) {
    state_property_step(c);
    performance_start(performance_ptr);
    gold_diffuse(c->prev, c->cur, DIFFUSION_RATE);
    performance_record(performance_ptr, step, DIFFUSE_TAG|COLOR_TAG);
  }

  if (USE_DENSITY_ADVECT) {
    state_property_step(c);
    performance_start(performance_ptr);
    gold_advect(c->prev, c->cur, x->cur, y->cur);
    performance_record(performance_ptr, step, ADVECT_TAG|COLOR_TAG);
  }

  // velocity
  if (USE_SOURCE_VELOCITIES)
    gold_source_velocities(x->prev, y->prev, x->cur, y->cur, step);

  if (USE_SINK_VELOCITIES)
    gold_sink_velocities(x->prev, y->prev, x->cur, y->cur);

  if (USE_VELOCITY_DIFFUSE) {

    state_property_step(x);
    performance_start(performance_ptr);
    gold_diffuse(x->prev, x->cur, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    state_property_step(y);
    performance_start(performance_ptr);
    gold_diffuse(y->prev, y->cur, VISCOSITY);
    performance_record(performance_ptr, step, DIFFUSE_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_project(x->cur, y->cur, p, d);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }
  if (USE_VELOCITY_ADVECT) {

    state_property_step(x);
    state_property_step(y);
    performance_start(performance_ptr);
    gold_advect(x->prev, x->cur, x->prev, y->prev);
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_advect(y->prev, y->cur, x->prev, y->prev);
    performance_record(performance_ptr, step, ADVECT_TAG|VELOCITY_TAG);

    performance_start(performance_ptr);
    gold_project(x->cur, y->cur, p, d);
    performance_record(performance_ptr, step, PROJECT_TAG);
  }
  free(p);
  free(d);
  performance_free(performance_ptr);
}
