#include <gold/index.h>
#include <util/compile_options.h>
#include <util/macros.h>
#include <util/state.h>
#include <util/idx2.cuh>
#include <stdio.h>
#include <util/vec2.cuh>
#include <kernel/index.cuh>

float _vxs0[N];
float _vxs[N];
float _vys0[N];
float _vys[N];
float _cs0[N];
float _cs[N];
float _pressures[N];
float _divergences[N];

void init() {
  previous_x_velocities = _vxs0;
  x_velocities = _vxs;
  previous_y_velocities = _vys0;
  y_velocities = _vys;
  previous_colors = _cs0;
  colors = _cs;
  pressures = _pressures;
  divergences = _divergences;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      colors[IDX2(idx)] = rand() / (float)RAND_MAX;
      previous_colors[IDX2(idx)] = colors[IDX2(idx)];
      x_velocities[IDX2(idx)] = 0.0;
      previous_x_velocities[IDX2(idx)] = 0.0;
      y_velocities[IDX2(idx)] = 0.0;
      previous_y_velocities[IDX2(idx)] = 0.0;
    }
  }
}

int main() {
  init();
  for (int i = 0; i < NUM_STEPS; i++) {
    if (i != 0)
      printf(",");
    for (int y = 0; y < HEIGHT; y++) {
      for (int x = 0; x < WIDTH; x++) {
        printf("%f", colors[y*WIDTH+x]);
        if (y != HEIGHT - 1 || x != WIDTH - 1)
          printf(",");
      }
    }
    gold_step(
      colors,
      previous_colors,
      previous_x_velocities,
      previous_y_velocities,
      x_velocities,
      y_velocities,
      pressures,
      divergences,
      current_step
    );
    // kernel_step_wrapper(
    //   colors,
    //   previous_colors,
    //   previous_x_velocities,
    //   previous_y_velocities,
    //   x_velocities,
    //   y_velocities,
    //   pressures,
    //   divergences,
    //   current_step
    // );
    current_step++;
  }
  return 0;
}