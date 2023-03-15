#include <config.h>

dim3 wrap_dim3(dim3 pos) {
  pos.x = (pos.x + WIDTH) % WIDTH;
  pos.y = (pos.y + HEIGHT) % HEIGHT;
  return pos;
}

int diffuse(float *xs0, float *xs) {
  for (int i = 0; i < N; i++) {
    dim3 pos = dim3(i % WIDTH, i / WIDTH, 0);
    dim3 neighbors[4] = {
      wrap_dim3(dim3(pos.x + 1, pos.y)),
      wrap_dim3(dim3(pos.x - 1, pos.y)),
      wrap_dim3(dim3(pos.x, pos.y + 1)),
      wrap_dim3(dim3(pos.x, pos.y - 1))
    };
  }
}