#include "gold_diffuse.h"
#include <macros.h>
#include <compile_options.h>
#include <idx2.h>
#include <assert.h>


void gold_diffuse(float *previous_values, float *values, float rate) {
  int *counts;
  float factor = TIME_STEP*rate*WIDTH*HEIGHT;
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        float sum = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          idx2 neighbor_idx = idx2_wrap(idx2(
              idx.x + adjancent_offsets[i].x,
              idx.y + adjancent_offsets[i].y
            ));
          sum += values[IDX2(neighbor_idx)];
        }
        values[IDX2(idx)] = (previous_values[IDX2(idx)] + factor*sum) / (1 + 4*factor);
      }
    }
  }
}