#include "../../include/resize.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

FloatImage resize_bilinear_omp(const FloatImage &input, int target_width, int target_height)
{
  if(input.width <= 0 || input.height <= 0 || input.channels <= 0 || target_width <= 0 || target_height <= 0){
    throw std::invalid_argument("image and target dimensions must be positive");
  }

  FloatImage output{target_width, target_height, input.channels,
                    std::vector<float>(static_cast<std::size_t>(target_width) * target_height * input.channels)};

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < target_height; ++y){
    for(int x = 0; x < target_width; ++x){
      const float source_y = target_height == 1 ? 0.0f :
        static_cast<float>(y) * (input.height - 1) / (target_height - 1);
      const float source_x = target_width == 1 ? 0.0f :
        static_cast<float>(x) * (input.width - 1) / (target_width - 1);
      const int y0 = static_cast<int>(std::floor(source_y));
      const int y1 = std::min(y0 + 1, input.height - 1);
      const int x0 = static_cast<int>(std::floor(source_x));
      const int x1 = std::min(x0 + 1, input.width - 1);
      const float fy = source_y - y0;
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
