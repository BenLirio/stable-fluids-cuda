#include "gold.h"
#include "macros.h"
#include "compile_options.h"
#include "gold_advect.h"
#include "gold_diffuse.h"
#include "gold_project.h"
#include "vec2.h"
#include "idx2.h"
#include "state.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


void source_colors() {
  vec2 center = vec2(WIDTH/2.0, HEIGHT/2.0);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2((float)x, (float)y);
      float distance = fmax(vec2_dist(center, position), 0.01);
      float magnitude = 0.01*((4+(rand()/(float)RAND_MAX))/(distance*distance));
      colors[IDX2(idx)] += magnitude;
      previous_colors[IDX2(idx)] += magnitude;
    }
  }
}
void sink_colors() {
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      colors[IDX2(idx)] *= (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
      previous_colors[IDX2(idx)] *= (1-TIME_STEP) + (1-COLOR_SINK_RATE)*TIME_STEP;
    }
  }
}
void source_velocity() {
  vec2 center = vec2(WIDTH/2.0, HEIGHT/2.0);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 position = vec2((float)x, (float)y);
      float distance = fmax(vec2_dist(center, position), 0.01);
      float magnitude = 0.01*((4+(rand()/(float)RAND_MAX))/(distance*distance));
      float percent_complete = (float)current_step / (float)NUM_STEPS;
      float x_magnitude = magnitude*cos(percent_complete*M_PI*10.0);
      float y_magnitude = magnitude*sin(percent_complete*M_PI*10.0);
      x_velocities[IDX2(idx)] += x_magnitude;
      previous_x_velocities[IDX2(idx)] += x_magnitude;
      y_velocities[IDX2(idx)] += y_magnitude;
      previous_y_velocities[IDX2(idx)] += y_magnitude;
    }
  }
}
void sink_velocity() {
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      previous_x_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      y_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
      previous_y_velocities[IDX2(idx)] *= (1-TIME_STEP) + (1-VELOCITY_SINK_RATE)*TIME_STEP;
    }
  }
}

void step() {
  // density
  source_colors();
  sink_colors();
  SWAP(previous_colors, colors);
  gold_diffuse(previous_colors, colors, DIFFUSION_RATE);
  SWAP(previous_colors, colors);
  gold_advect(previous_colors, colors, x_velocities, y_velocities);

  // velocity
  source_velocity();
  // sink_velocity();
  SWAP(previous_x_velocities, x_velocities);
  gold_diffuse(previous_x_velocities, x_velocities, VISCOSITY);
  SWAP(previous_y_velocities, y_velocities);
  gold_diffuse(previous_y_velocities, y_velocities, VISCOSITY);
  gold_project(x_velocities, y_velocities, preasure, divergence);

  SWAP(previous_x_velocities, x_velocities);
  SWAP(previous_y_velocities, y_velocities);
  gold_advect(previous_x_velocities, x_velocities, previous_x_velocities, previous_y_velocities);
  gold_advect(previous_y_velocities, y_velocities, previous_x_velocities, previous_y_velocities);
  gold_project(previous_x_velocities, previous_y_velocities, preasure, divergence);
  current_step++;
}
