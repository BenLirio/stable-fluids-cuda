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

void gold_step(
  float *previous_colors_pointer,
  float *colors_pointer,
  float *previous_x_velocities,
  float *previous_y_velocities,
  float *x_velocities,
  float *y_velocities,
  float *pressures,
  float *divergences,
  int current_step
) {
  // density
  // gold_source_colors(previous_colors, colors);
  // gold_sink_colors(previous_colors, colors);
  SWAP(previous_colors, colors);
  gold_diffuse(previous_colors, colors, DIFFUSION_RATE);
  // SWAP(previous_colors, colors);
  // gold_advect(previous_colors, colors, x_velocities, y_velocities);

  // // velocity
  // gold_source_velocities(previous_x_velocities, previous_y_velocities, x_velocities, y_velocities, current_step);
  // gold_sink_velocities(previous_x_velocities, previous_y_velocities, x_velocities, y_velocities);
  // SWAP(previous_x_velocities, x_velocities);
  // gold_diffuse(previous_x_velocities, x_velocities, VISCOSITY);
  // SWAP(previous_y_velocities, y_velocities);
  // gold_diffuse(previous_y_velocities, y_velocities, VISCOSITY);
  // gold_project(x_velocities, y_velocities, pressures, divergences);

  // SWAP(previous_x_velocities, x_velocities);
  // SWAP(previous_y_velocities, y_velocities);
  // gold_advect(previous_x_velocities, x_velocities, previous_x_velocities, previous_y_velocities);
  // gold_advect(previous_y_velocities, y_velocities, previous_x_velocities, previous_y_velocities);
  // gold_project(x_velocities, y_velocities, pressures, divergences);
}
