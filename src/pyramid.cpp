#include "../include/pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

std::size_t FloatImage::size() const
{
  return static_cast<std::size_t>(width) * height * channels;
}

namespace {

int clamp_coordinate(int coordinate, int limit)
{
  return std::clamp(coordinate, 0, limit - 1);
}

FloatImage blur_float(const FloatImage &input, const std::vector<float> &weights)
{
  if(input.width <= 0 || input.height <= 0 || input.channels <= 0 || weights.empty() || weights.size() % 2 == 0){
    throw std::invalid_argument("invalid image or Gaussian weights");
  }

  const int radius = static_cast<int>(weights.size() / 2);
  FloatImage horizontal{input.width, input.height, input.channels, std::vector<float>(input.size())};
  FloatImage output{input.width, input.height, input.channels, std::vector<float>(input.size())};

  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      const std::size_t output_index = (static_cast<std::size_t>(y) * input.width + x) * input.channels;
      for(int kx = -radius; kx <= radius; ++kx){
        const int nx = clamp_coordinate(x + kx, input.width);
        const std::size_t input_index = (static_cast<std::size_t>(y) * input.width + nx) * input.channels;
        const float weight = weights[kx + radius];
        for(int channel = 0; channel < input.channels; ++channel){
          horizontal.pixels[output_index + channel] += weight * input.pixels[input_index + channel];
        }
      }
    }
  }

  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      const std::size_t output_index = (static_cast<std::size_t>(y) * input.width + x) * input.channels;
      for(int ky = -radius; ky <= radius; ++ky){
        const int ny = clamp_coordinate(y + ky, input.height);
        const std::size_t input_index = (static_cast<std::size_t>(ny) * input.width + x) * input.channels;
        const float weight = weights[ky + radius];
        for(int channel = 0; channel < input.channels; ++channel){
          output.pixels[output_index + channel] += weight * horizontal.pixels[input_index + channel];
        }
      }
    }
  }

  return output;
}

}

FloatImage downsample(const FloatImage &input, const std::vector<float> &weights)
{
  const FloatImage blurred = blur_float(input, weights);
  const int output_width = (input.width + 1) / 2;
  const int output_height = (input.height + 1) / 2;
  FloatImage output{output_width, output_height, input.channels,
                    std::vector<float>(static_cast<std::size_t>(output_width) * output_height * input.channels)};

  for(int y = 0; y < output.height; ++y){
    for(int x = 0; x < output.width; ++x){
      const std::size_t source_index = (static_cast<std::size_t>(2 * y) * input.width + 2 * x) * input.channels;
      const std::size_t output_index = (static_cast<std::size_t>(y) * output.width + x) * output.channels;
      for(int channel = 0; channel < input.channels; ++channel){
        output.pixels[output_index + channel] = blurred.pixels[source_index + channel];
      }
    }
  }

  return output;
}

FloatImage expand_bilinear(const FloatImage &input, int target_width, int target_height)
{
  if(input.width <= 0 || input.height <= 0 || target_width <= 0 || target_height <= 0){
    throw std::invalid_argument("image dimensions must be positive");
  }

  FloatImage output{target_width, target_height, input.channels,
                    std::vector<float>(static_cast<std::size_t>(target_width) * target_height * input.channels)};

  for(int y = 0; y < target_height; ++y){
    const float source_y = target_height == 1 ? 0.0f :
      static_cast<float>(y) * (input.height - 1) / (target_height - 1);
    const int y0 = static_cast<int>(std::floor(source_y));
    const int y1 = std::min(y0 + 1, input.height - 1);
    const float fy = source_y - y0;

    for(int x = 0; x < target_width; ++x){
      const float source_x = target_width == 1 ? 0.0f :
        static_cast<float>(x) * (input.width - 1) / (target_width - 1);
      const int x0 = static_cast<int>(std::floor(source_x));
      const int x1 = std::min(x0 + 1, input.width - 1);
      const float fx = source_x - x0;
      const std::size_t output_index = (static_cast<std::size_t>(y) * target_width + x) * output.channels;

      for(int channel = 0; channel < input.channels; ++channel){
        const float top_left = input.pixels[(static_cast<std::size_t>(y0) * input.width + x0) * input.channels + channel];
        const float top_right = input.pixels[(static_cast<std::size_t>(y0) * input.width + x1) * input.channels + channel];
        const float bottom_left = input.pixels[(static_cast<std::size_t>(y1) * input.width + x0) * input.channels + channel];
        const float bottom_right = input.pixels[(static_cast<std::size_t>(y1) * input.width + x1) * input.channels + channel];
        const float top = top_left + fx * (top_right - top_left);
        const float bottom = bottom_left + fx * (bottom_right - bottom_left);
        output.pixels[output_index + channel] = top + fy * (bottom - top);
      }
    }
  }

  return output;
}
