#include "gold.h"
#include "config.h"
#include "vec2.h"
#include "state.h"

idx2 diffusion_offsets[NUM_NEIGHBORS] = {
    idx2(0, 1),
    idx2(1, 0),
    idx2(0, -1),
    idx2(-1, 0),
};
idx2 advection_offsets[NUM_NEIGHBORS] = {
  idx2(0, 0),
  idx2(0, 1),
  idx2(1, 0),
  idx2(1, 1),
};


void diffuse(float *xs0, float *xs) {
  float factor = TIME_STEP*DIFFUSION_RATE*WIDTH*HEIGHT;
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 0; y < HEIGHT; y++) {
      for (int x = 0; x < WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          sum += xs[IDX2(
            wrap_idx2(idx2(
              idx.x + diffusion_offsets[i].x,
              idx.y + diffusion_offsets[i].y
            )))];
        }
        xs[IDX2(idx)] = (xs0[IDX2(idx)] + factor*sum) / (1 + 4*factor);
      }
    }
  }
}

void advect(float *xs0, float *xs, float *vxs, float *vys) {
  float alpha = TIME_STEP*WIDTH*HEIGHT;
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      vec2 pos_offset_by_velocity = vec2(
        x - alpha*vxs[IDX2(idx2(x, y))],
        y - alpha*vys[IDX2(idx2(x, y))]
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
      xs[IDX2(idx2(x, y))] = 0;
      for (int i = 0; i < NUM_NEIGHBORS; i++) {
        float weight = weights[i];
        idx2 neighbor = wrap_idx2(idx2(
          idx_offset_by_velocity.x + advection_offsets[i].x,
          idx_offset_by_velocity.y + advection_offsets[i].y
        ));
        xs[IDX2(idx2(x, y))] += weight*xs0[IDX2(neighbor)];
      }
    }
  }
}



void step() {
  // diffusion
  SWAP(cs0, cs);
  diffuse(cs0, cs);
  SWAP(vxs0, vxs);
  diffuse(vxs0, vxs);
  SWAP(vys0, vys);
  diffuse(vys0, vys);

  // advection
  SWAP(cs0, cs);
  advect(cs0, cs, vxs, vys);
}
