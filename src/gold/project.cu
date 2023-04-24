#include <gold/project.h>

#include <util/macros.h>
#include <util/idx2.cuh>
#include <util/vec2.cuh>
#include <util/derivative.cuh>


void gold_project(float *x_velocities, float *y_velocities, float *pressures, float *divergence) {

  float h = 1.0f / sqrt(N);

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      float x_velocity_derivative = get_x_derivative(x_velocities, idx);
      float y_velocity_derivative = get_y_derivative(y_velocities, idx);
      divergence[IDX2(idx)] = -h * (x_velocity_derivative + y_velocity_derivative)/2;
      pressures[IDX2(idx)] = 0.0f;
    }
  }

  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          idx2 neighbor_idx = idx2_add(idx, adjancent_offsets[i]);
          sum += pressures[IDX2(neighbor_idx)];
        }
        pressures[IDX2(idx)] = (divergence[IDX2(idx)] + sum) / 4;
      }
    }
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] -= get_x_derivative(pressures, idx) / (2*h);
      y_velocities[IDX2(idx)] -= get_y_derivative(pressures, idx) / (2*h);
    }
  }
}