#include "gold.h"
#include "config.h"
#include "vec2.h"
#include "state.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

idx2 adjancent_offsets[NUM_NEIGHBORS] = {
    idx2(0, 1),
    idx2(1, 0),
    idx2(0, -1),
    idx2(-1, 0),
};
idx2 lower_right_square_offsets[NUM_NEIGHBORS] = {
  idx2(0, 0),
  idx2(0, 1),
  idx2(1, 0),
  idx2(1, 1),
};


void diffuse(float *previous_values, float *values, float rate) {
  int *counts;
  if (ASSERTIONS_ENABLED) {
    counts = (int*)malloc(WIDTH*HEIGHT*sizeof(int));
    for (int i = 0; i < WIDTH*HEIGHT; i++) {
      counts[i] = 0;
    }
  }

  float factor = TIME_STEP*rate*WIDTH*HEIGHT;
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          idx2 neighbor_idx = wrap_idx2(idx2(
              idx.x + adjancent_offsets[i].x,
              idx.y + adjancent_offsets[i].y
            ));
          sum += values[IDX2(neighbor_idx)];
          if (ASSERTIONS_ENABLED) counts[IDX2(neighbor_idx)]++;
        }
        values[IDX2(idx)] = (previous_values[IDX2(idx)] + factor*sum) / (1 + 4*factor);
      }
    }
  }
  if (ASSERTIONS_ENABLED) {
    for (int i = 0; i < WIDTH*HEIGHT; i++) {
      assert(counts[i] == 4*GAUSS_SEIDEL_ITERATIONS);
    }
    free(counts);
  }
}

void advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = TIME_STEP*sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 pos_offset_by_velocity = wrap_vec2(vec2(
        x - alpha*x_velocities[IDX2(idx)],
        y - alpha*y_velocities[IDX2(idx)]
      ));
      idx2 idx_offset_by_velocity = wrap_idx2(idx2(
        (int)floor(pos_offset_by_velocity.x),
        (int)floor(pos_offset_by_velocity.y)
      ));
      float wx0 = min(min(
        abs(pos_offset_by_velocity.x - idx_offset_by_velocity.x),
        abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x + WIDTH))
      ), abs(pos_offset_by_velocity.x - (idx_offset_by_velocity.x - WIDTH)));
      if (ASSERTIONS_ENABLED && VERBOSE_ASSERTIONS && (wx0 < 0.0f || wx0 > 1.0f)) {
        fprintf(stderr, "advect: y=%d; x=%d; pos=%f; idx=%d; wx0=%f;\n", y, x, pos_offset_by_velocity.x, idx_offset_by_velocity.x, wx0);
      }
      if (ASSERTIONS_ENABLED) assert(wx0 >= 0.0f && wx0 <= 1.0f);
      float wx1 = 1 - wx0;
      if (ASSERTIONS_ENABLED) assert(wx1 >= 0.0f && wx1 <= 1.0f);
      //float wy0 = pos_offset_by_velocity.y - idx_offset_by_velocity.y;
      float wy0 = min(min(
        abs(pos_offset_by_velocity.y - idx_offset_by_velocity.y),
        abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y + HEIGHT))
      ), abs(pos_offset_by_velocity.y - (idx_offset_by_velocity.y - HEIGHT)));
      if (ASSERTIONS_ENABLED) assert(wy0 >= 0.0f && wy0 <= 1.0f);
      float wy1 = 1 - wy0;
      if (ASSERTIONS_ENABLED) assert(wy1 >= 0.0f && wy1 <= 1.0f);
      float weights[NUM_NEIGHBORS] = {
        wx1*wy1,
        wx0*wy1,
        wx1*wy0,
        wx0*wy0,
      };
      values[IDX2(idx)] = 0;
      for (int i = 0; i < NUM_NEIGHBORS; i++) {
        float weight = weights[i];
        idx2 neighbor = wrap_idx2(idx2(
          idx_offset_by_velocity.x + lower_right_square_offsets[i].x,
          idx_offset_by_velocity.y + lower_right_square_offsets[i].y
        ));
        values[IDX2(idx)] += weight*previous_values[IDX2(neighbor)];
      }
    }
  }
}

void project(float *x_velocities, float *y_velocities, float *preasure, float *divergence) {
  float h = 1.0f / sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      float x_velocity_derivative = x_velocities[IDX2(wrap_idx2(idx2(x+1, y)))] - x_velocities[IDX2(wrap_idx2(idx2(x-1, y)))];
      float y_velocity_derivative = y_velocities[IDX2(wrap_idx2(idx2(x, y+1)))] - y_velocities[IDX2(wrap_idx2(idx2(x, y-1)))];
      divergence[IDX2(idx)] = -0.5f * h * (x_velocity_derivative + y_velocity_derivative);
      preasure[IDX2(idx)] = 0;
    }
  }

  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          sum += preasure[IDX2(wrap_idx2(idx2(
            idx.x + adjancent_offsets[i].x,
            idx.y + adjancent_offsets[i].y
          )))];
        }
        preasure[IDX2(idx)] = (divergence[IDX2(idx)] + sum) / 4;
      }
    }
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] -= 0.5f * (preasure[IDX2(wrap_idx2(idx2(x+1, y)))] - preasure[IDX2(wrap_idx2(idx2(x-1, y)))]) / h;
      y_velocities[IDX2(idx)] -= 0.5f * (preasure[IDX2(wrap_idx2(idx2(x, y+1)))] - preasure[IDX2(wrap_idx2(idx2(x, y-1)))]) / h;
    }
  }
}

void source_colors() {
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      if (x > WIDTH/2 - WIDTH*0.1 && x < WIDTH/2 + WIDTH*0.1 && y > HEIGHT/2 - HEIGHT*0.1 && y < HEIGHT/2 + HEIGHT*0.1) {
        colors[IDX2(idx)] += 0.01;
      }
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
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      y_velocities[IDX2(idx)] += 0.1*fabs(sin(x*0.1 + y*0.1 + current_step/(float)NUM_STEPS));
      previous_y_velocities[IDX2(idx)] += 0.1*fabs(sin(x*0.1 + y*0.1 + current_step/(float)NUM_STEPS));
      x_velocities[IDX2(idx)] += 0.1*fabs(cos(x*0.1 + y*0.1 + current_step/(float)NUM_STEPS));
      previous_x_velocities[IDX2(idx)] += 0.1*fabs(cos(x*0.1 + y*0.1 + current_step/(float)NUM_STEPS));

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
  diffuse(previous_colors, colors, DIFFUSION_RATE);
  SWAP(previous_colors, colors);
  advect(previous_colors, colors, x_velocities, y_velocities);

  // velocity
  source_velocity();
  sink_velocity();
  SWAP(previous_x_velocities, x_velocities);
  diffuse(previous_x_velocities, x_velocities, VISCOSITY);
  SWAP(previous_y_velocities, y_velocities);
  diffuse(previous_y_velocities, y_velocities, VISCOSITY);
  project(x_velocities, y_velocities, preasure, divergence);

  SWAP(previous_x_velocities, x_velocities);
  SWAP(previous_y_velocities, y_velocities);
  advect(previous_x_velocities, x_velocities, previous_x_velocities, previous_y_velocities);
  advect(previous_y_velocities, y_velocities, previous_x_velocities, previous_y_velocities);
  project(previous_x_velocities, previous_y_velocities, preasure, divergence);
  current_step++;
}
