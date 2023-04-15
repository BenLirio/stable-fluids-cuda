#include <gtest/gtest.h>
#include <util/idx2.cuh>
#include <util/vec2.h>
#include <util/compile_options.h>
#include <math.h>
#include <util/macros.h>

TEST(VEC2, wrap_0) {
  vec2 u = vec2_wrap(vec2(WIDTH, HEIGHT));
  EXPECT_FLOAT_EQ(u.x, WIDTH);
  EXPECT_FLOAT_EQ(u.y, HEIGHT);
}
TEST(VEC2, wrap_1) {
  vec2 u = vec2_wrap(vec2(WIDTH+1.0, HEIGHT+1.0));
  EXPECT_FLOAT_EQ(u.x, 1.0);
  EXPECT_FLOAT_EQ(u.y, 1.0);
}
TEST(VEC2, wrap_2) {
  vec2 u = vec2_wrap(vec2(WIDTH+0.501, HEIGHT+0.501));
  EXPECT_NEAR(u.x, 0.501, EQ_THRESHOLD);  
  EXPECT_NEAR(u.y, 0.501, EQ_THRESHOLD);  
}
TEST(VEC2, wrap_3) {
  vec2 u = vec2_wrap(vec2(0.0, 0.0));
  EXPECT_FLOAT_EQ(u.x, (float)WIDTH);
  EXPECT_FLOAT_EQ(u.y, (float)HEIGHT);
}
TEST(VEC2, dist_0) {
  vec2 u = vec2_wrap(vec2(0.0, 0.0));
  vec2 v = vec2_wrap(vec2(1.0, 1.0));
  EXPECT_NEAR(vec2_dist(u, v), sqrt(2.0), EQ_THRESHOLD);
}
TEST(VEC2, dist_1) {
  vec2 u = vec2_wrap(vec2(WIDTH, HEIGHT));
  vec2 v = vec2_wrap(vec2(1.0, 1.0));
  EXPECT_NEAR(vec2_dist(u, v), sqrt(2.0), EQ_THRESHOLD);
}
TEST(VEC2, dist_2) {
  vec2 u = vec2_wrap(vec2(WIDTH+0.5, HEIGHT));
  vec2 v = vec2_wrap(vec2(0.5, 1.0));
  EXPECT_NEAR(vec2_dist(u, v), 1.0, EQ_THRESHOLD);
}