#ifndef STABLE_FLUIDS_CUDA_GOLD_PROJECT_H_
#define STABLE_FLUIDS_CUDA_GOLD_PROJECT_H_

#include <util/idx2.cuh>

void gold_project(float *x_velocities, float *y_velocities, float *pressure, float *divergence);

#endif // STABLE_FLUIDS_CUDA_GOLD_PROJECT_H_