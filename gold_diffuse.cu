#include "gold_diffuse.h"
#include "macros.h"
#include "compile_options.h"
#include "idx2.h"
#include <assert.h>


void gold_diffuse(float *previous_values, float *values, float rate) {
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