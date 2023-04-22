#ifndef STABLE_FLUIDS_CUDA_GOLD_H
#define STABLE_FLUIDS_CUDA_GOLD_H

void gold_step(
  float *previous_colors_pointer,
  float *colors_pointer,
  float *previous_x_velocities,
  float *previous_y_velocities,
  float *x_velocities,
  float *y_velocities,
  float *pressures,
  float *divergences,
  int current_step
);

#endif // STABLE_FLUIDS_CUDA_GOLD_H
