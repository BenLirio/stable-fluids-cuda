#include <gold/project.h>
#include <util/compile_options.h>
#include <util/macros.h>
#include <util/idx2.h>
#include <util/vec2.h>

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
      float x_velocity_derivative = get_x_derivative(x_velocities, idx);
      float y_velocity_derivative = get_y_derivative(y_velocities, idx);
      divergence[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
      pressure[IDX2(idx)] = 0.0;
    }
  }

  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          idx2 neighbor_idx = idx2_add(idx, adjancent_offsets[i]);
          sum += pressure[IDX2(neighbor_idx)];
        }
        pressure[IDX2(idx)] = (divergence[IDX2(idx)] + sum) / 4;
      }
    }
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] -= get_x_derivative(pressure, idx) / (2*h);
      y_velocities[IDX2(idx)] -= get_y_derivative(pressure, idx) / (2*h);
    }
  }
}