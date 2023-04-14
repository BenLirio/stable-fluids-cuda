#include "gold_project.h"
#include "compile_options.h"
#include "macros.h"
#include "idx2.h"
#include "vec2.h"

float get_x_derivative(float *x_velocities, idx2 idx) {
  idx2 next_idx = idx2_add(idx, idx2(1, 0));
  idx2 previous_idx = idx2_add(idx, idx2(-1, 0));
  return x_velocities[IDX2(next_idx)] - x_velocities[IDX2(previous_idx)];
}

float get_y_derivative(float *y_velocities, idx2 idx) {
  idx2 next_idx = idx2_add(idx, idx2(0, 1));
  idx2 previous_idx = idx2_add(idx, idx2(0, -1));
  return y_velocities[IDX2(next_idx)] - y_velocities[IDX2(previous_idx)];
}

void gold_project(float *x_velocities, float *y_velocities, float *pressure, float *divergence) {
  float h = 1.0f / sqrt(N);
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      float x_velocity_derivative = x_velocities[IDX2(idx2_wrap(idx2(x+1, y)))] - x_velocities[IDX2(idx2_wrap(idx2(x-1, y)))];
      float y_velocity_derivative = y_velocities[IDX2(idx2_wrap(idx2(x, y+1)))] - y_velocities[IDX2(idx2_wrap(idx2(x, y-1)))];
      divergence[IDX2(idx)] = -0.5f * h * (x_velocity_derivative + y_velocity_derivative);
      pressure[IDX2(idx)] = 0;
    }
  }

  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          sum += pressure[IDX2(idx2_wrap(idx2(
            idx.x + adjancent_offsets[i].x,
            idx.y + adjancent_offsets[i].y
          )))];
        }
        pressure[IDX2(idx)] = (divergence[IDX2(idx)] + sum) / 4;
      }
    }
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] -= 0.5f * (pressure[IDX2(idx2_wrap(idx2(x+1, y)))] - pressure[IDX2(idx2_wrap(idx2(x-1, y)))]) / h;
      y_velocities[IDX2(idx)] -= 0.5f * (pressure[IDX2(idx2_wrap(idx2(x, y+1)))] - pressure[IDX2(idx2_wrap(idx2(x, y-1)))]) / h;
    }
  }
}