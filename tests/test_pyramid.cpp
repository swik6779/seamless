#include "../include/pyramid.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  FloatImage input{5, 3, 1, {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    10.0f, 11.0f, 12.0f, 13.0f, 14.0f
  }};

  const std::vector<float> identity_weights{1.0f};
  const FloatImage reduced = downsample(input, identity_weights);
  assert(reduced.width == 3);
  assert(reduced.height == 2);
  assert(reduced.pixels[0] == 0.0f);
  assert(reduced.pixels[1] == 2.0f);
  assert(reduced.pixels[2] == 4.0f);
  assert(reduced.pixels[3] == 10.0f);
  assert(reduced.pixels[4] == 12.0f);
  assert(reduced.pixels[5] == 14.0f);

  const FloatImage expanded = expand_bilinear(reduced, 5, 3);
  assert(expanded.width == 5);
  assert(expanded.height == 3);
  assert(expanded.channels == 1);
  assert(std::fabs(expanded.pixels.front() - 0.0f) < 1e-6f);
  assert(std::fabs(expanded.pixels.back() - 14.0f) < 1e-6f);

  const FloatImage single_pixel = expand_bilinear(FloatImage{1, 1, 1, {3.5f}}, 4, 3);
  for(float value : single_pixel.pixels){
    assert(std::fabs(value - 3.5f) < 1e-6f);
  }

  std::cout << "pyramid primitive tests passed\n";
  return 0;
}
