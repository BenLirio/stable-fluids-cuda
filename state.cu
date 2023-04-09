#include "state.h"

float _vxs0[N];
float _vxs[N];
float _vys0[N];
float _vys[N];
float _cs0[N];
float _cs[N];

float *previous_x_velocities = _vxs0;
float *x_velocities = _vxs;
float *previous_y_velocities = _vys0;
float *y_velocities = _vys;
float *previous_colors = _cs0;
float *colors = _cs;