#ifndef STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_
#define STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_

void kernel_step(
  float *colors,
  float *previous_colors,
  float *x_velocities,
  float *previous_x_velocities,
  float *y_velocities,
  float *previous_y_velocities,
  float *preasures,
  float *divergences
);
void kernel_step_wrapper(
  float *colors,
  float *previous_colors,
  float *x_velocities,
  float *previous_x_velocities,
  float *y_velocities,
  float *previous_y_velocities,
  float *preasures,
  float *divergences
);

#endif // STABLE_FLUIDS_CUDA_KERNEL_INDEX_H_