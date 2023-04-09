#include "gold.h"
#include "config.h"
#include "vec2.h"
#include "state.h"

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
  float factor = TIME_STEP*rate*WIDTH*HEIGHT;
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          sum += values[IDX2(
            wrap_idx2(idx2(
              idx.x + adjancent_offsets[i].x,
              idx.y + adjancent_offsets[i].y
            )))];
        }
        values[IDX2(idx)] = (previous_values[IDX2(idx)] + factor*sum) / (1 + 4*factor);
      }
    }
  }
}

void advect(float *previous_values, float *values, float *x_velocities, float *y_velocities) {
  float alpha = (TIME_STEP*WIDTH*HEIGHT)/(2.0f*(WIDTH+HEIGHT));
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      vec2 pos_offset_by_velocity = vec2(
        x - alpha*x_velocities[IDX2(idx2(x, y))],
        y - alpha*y_velocities[IDX2(idx2(x, y))]
      );
      idx2 idx_offset_by_velocity = idx2(
        (int)floor(pos_offset_by_velocity.x),
        (int)floor(pos_offset_by_velocity.y)
      );
      float wx0 = idx_offset_by_velocity.x - pos_offset_by_velocity.x;
      float wx1 = 1 - wx0;
      float wy0 = idx_offset_by_velocity.y - pos_offset_by_velocity.y;
      float wy1 = 1 - wy0;
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

void project(float *x_velocities, float *y_velocities, float *previous_x_velocities, float *previous_y_velocities) {
  float h = 1.0f / N;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      float delta_vx = x_velocities[IDX2(idx2(x+1, y))] - x_velocities[IDX2(idx2(x-1, y))];
      float delta_vy = y_velocities[IDX2(idx2(x, y+1))] - y_velocities[IDX2(idx2(x, y-1))];
      previous_y_velocities[IDX2(idx)] = -0.5f * h * (delta_vx + delta_vy);
      previous_x_velocities[IDX2(idx)] = 0;
    }
  }

  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          sum += previous_y_velocities[IDX2(
            wrap_idx2(idx2(
              idx.x + adjancent_offsets[i].x,
              idx.y + adjancent_offsets[i].y
            )))];
        }
        previous_y_velocities[IDX2(idx)] = (previous_y_velocities[IDX2(idx)] + sum) / 4;
      }
    }
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      x_velocities[IDX2(idx)] -= 0.5f * (previous_y_velocities[IDX2(idx2(x+1, y))] - previous_y_velocities[IDX2(idx2(x-1, y))]) / h;
      y_velocities[IDX2(idx)] -= 0.5f * (previous_y_velocities[IDX2(idx2(x, y+1))] - previous_y_velocities[IDX2(idx2(x, y-1))]) / h;
    }
  }
}

void step() {
  // density
  SWAP(previous_colors, colors);
  diffuse(previous_colors, colors, DIFFUSION_RATE);
  SWAP(previous_colors, colors);
  advect(previous_colors, colors, x_velocities, y_velocities);

  // velocity
  SWAP(previous_x_velocities, x_velocities);
  diffuse(previous_x_velocities, x_velocities, VISCOSITY);
  SWAP(previous_y_velocities, y_velocities);
  diffuse(previous_y_velocities, y_velocities, VISCOSITY);
  project(previous_x_velocities, previous_y_velocities, x_velocities, y_velocities);

  SWAP(previous_x_velocities, x_velocities);
  SWAP(previous_y_velocities, y_velocities);
  advect(previous_x_velocities, x_velocities, previous_x_velocities, previous_y_velocities);
  advect(previous_y_velocities, y_velocities, previous_x_velocities, previous_y_velocities);
  project(previous_x_velocities, previous_y_velocities, x_velocities, y_velocities);
}
