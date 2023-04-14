#include <gtest/gtest.h>
#include "idx2.h"
#include "vec2.h"
#include "gold.h"
#include "compile_options.h"
#include <math.h>
#include "macros.h"

TEST(IDX2, wrap_0) {
  idx2 u = wrap_idx2(idx2(WIDTH, HEIGHT));
  EXPECT_EQ(u.x, WIDTH);
  EXPECT_EQ(u.y, HEIGHT);
}
TEST(IDX2, wrap_1) {
  idx2 u = wrap_idx2(idx2(WIDTH+1, HEIGHT+1));
  EXPECT_EQ(u.x, 1);
  EXPECT_EQ(u.y, 1);
}
TEST(IDX2, wrap_2) {
  idx2 u = wrap_idx2(idx2(0, 0));
  EXPECT_EQ(u.x, WIDTH);
  EXPECT_EQ(u.y, HEIGHT);
}
TEST(IDX2, wrap_3) {
  idx2 u = wrap_idx2(idx2(-1, -1));
  EXPECT_EQ(u.x, WIDTH-1);
  EXPECT_EQ(u.y, HEIGHT-1);
}