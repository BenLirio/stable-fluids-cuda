#ifndef STABLE_FLUIDS_CUDA_STATE_H
#define STABLE_FLUIDS_CUDA_STATE_H

extern float *previous_x_velocities;
extern float *x_velocities;
extern float *previous_y_velocities;
extern float *y_velocities;
extern float *previous_colors;
extern float *colors;
extern float *preasure;
extern float *divergence;
extern int current_step;

#endif // STABLE_FLUIDS_CUDA_STATE_H