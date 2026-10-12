#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include <cstddef>
#include <vector>

struct FloatImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> pixels;

  std::size_t size() const;
};

// All coordinates use clamp-to-edge borders and row-major interleaved channels.
FloatImage downsample(const FloatImage &input, const std::vector<float> &weights);
FloatImage expand_bilinear(const FloatImage &input, int target_width, int target_height);

#endif
