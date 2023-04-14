#include <gtest/gtest.h>
#include "idx2.h"
#include "vec2.h"
#include "gold.h"
#include "compile_options.h"
#include <math.h>
#include "macros.h"
#include "gold_diffuse.h"

TEST(Diffuse, disperses_evenly) {
  float _previous_color[N];
  float _color[N];
  float *previous_color = _previous_color;
  float *color = _color;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      color[IDX2(idx)] = x + y;
      previous_color[IDX2(idx)] = color[IDX2(idx)];
    }
  }

  for (int i = 0; i < 10000; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  float total = 0.0;
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      idx2 idx = idx2(x, y);
      total += color[IDX2(idx)];
    }
  }

  float average = total / (float)N;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      EXPECT_NEAR(color[IDX2(idx)], average, EQ_THRESHOLD);
    }
  }
}

TEST(Diffuse, zero_sum) {
  float _previous_color[N];
  float _color[N];
  float *previous_color = _previous_color;
  float *color = _color;

  float initial_total = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      color[IDX2(idx)] = rand()/RAND_MAX;
      previous_color[IDX2(idx)] = color[IDX2(idx)];
      initial_total += color[IDX2(idx)];
    }
  }

  for (int i = 0; i < 10000; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  float after_total = 0.0;
  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      after_total += color[IDX2(idx)];
    }
  }
  EXPECT_NEAR(initial_total, after_total, EQ_THRESHOLD);
}

TEST(Diffuse, single_cell) {
  float _previous_color[N];
  float _color[N];
  float *previous_color = _previous_color;
  float *color = _color;

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      if (x == 1 && y == 1) {
        color[IDX2(idx)] = (float)N;
      } else {
        color[IDX2(idx)] = 0.0;
      }
      previous_color[IDX2(idx)] = color[IDX2(idx)];
    }
  }

  for (int i = 0; i < 10000; i++) {
    SWAP(previous_color, color);
    gold_diffuse(previous_color, color, DIFFUSION_RATE);
  }

  for (int y = 1; y <= HEIGHT; y++) {
    for (int x = 1; x <= WIDTH; x++) {
      idx2 idx = idx2(x, y);
      EXPECT_NEAR(color[IDX2(idx)], 1.0, EQ_THRESHOLD);
    }
  }
}