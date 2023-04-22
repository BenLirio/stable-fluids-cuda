#include <gold/diffuse.h>
#include <util/macros.h>
#include <util/compile_options.h>
#include <util/idx2.cuh>
#include <assert.h>


void gold_diffuse(float *previous_values, float *values, float rate) {
  float factor = TIME_STEP*rate*N;
  for (int k = 0; k < GAUSS_SEIDEL_ITERATIONS; k++) {
    for (int y = 1; y <= HEIGHT; y++) {
      for (int x = 1; x <= WIDTH; x++) {
        idx2 idx = idx2(x, y);
        values[IDX2(idx)] = (
          previous_values[IDX2(idx)] +
          factor*(
            values[IDX2(idx2_add(idx, idx2(1, 0)))] +
            values[IDX2(idx2_add(idx, idx2(-1, 0)))] +
            values[IDX2(idx2_add(idx, idx2(0, 1)))] +
            values[IDX2(idx2_add(idx, idx2(0, -1)))]
          )
        ) / (1 + 4*factor);
      }
    }
  }
}