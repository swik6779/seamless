#include "../../include/pyramid_omp.hpp"
#include "../../include/resize.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace {

int clamp_coordinate(int coordinate, int limit)
{
  return std::clamp(coordinate, 0, limit - 1);
}

}

FloatImage blur_float_omp(const FloatImage &input, const std::vector<float> &weights)
{
  if(input.width <= 0 || input.height <= 0 || input.channels <= 0 || weights.empty() || weights.size() % 2 == 0){
    throw std::invalid_argument("invalid image or Gaussian weights");
  }

  const int radius = static_cast<int>(weights.size() / 2);
  FloatImage horizontal{input.width, input.height, input.channels, std::vector<float>(input.size())};
  FloatImage output{input.width, input.height, input.channels, std::vector<float>(input.size())};

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      const std::size_t output_index = (static_cast<std::size_t>(y) * input.width + x) * input.channels;
      for(int kx = -radius; kx <= radius; ++kx){
        const int nx = clamp_coordinate(x + kx, input.width);
        const std::size_t input_index = (static_cast<std::size_t>(y) * input.width + nx) * input.channels;
        for(int channel = 0; channel < input.channels; ++channel){
          horizontal.pixels[output_index + channel] += weights[kx + radius] * input.pixels[input_index + channel];
        }
      }
    }
  }

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      const std::size_t output_index = (static_cast<std::size_t>(y) * input.width + x) * input.channels;
      for(int ky = -radius; ky <= radius; ++ky){
        const int ny = clamp_coordinate(y + ky, input.height);
        const std::size_t input_index = (static_cast<std::size_t>(ny) * input.width + x) * input.channels;
        for(int channel = 0; channel < input.channels; ++channel){
          output.pixels[output_index + channel] += weights[ky + radius] * horizontal.pixels[input_index + channel];
        }
      }
    }
  }
  return output;
}

FloatImage downsample_from_blurred_omp(const FloatImage &blurred)
{
  if(blurred.width <= 0 || blurred.height <= 0 || blurred.channels <= 0){
    throw std::invalid_argument("invalid blurred image");
  }
  const int output_width = (blurred.width + 1) / 2;
  const int output_height = (blurred.height + 1) / 2;
  FloatImage output{output_width, output_height, blurred.channels,
                    std::vector<float>(static_cast<std::size_t>(output_width) * output_height * blurred.channels)};

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < output.height; ++y){
    for(int x = 0; x < output.width; ++x){
      const std::size_t source_index = (static_cast<std::size_t>(2 * y) * blurred.width + 2 * x) * blurred.channels;
      const std::size_t output_index = (static_cast<std::size_t>(y) * output.width + x) * output.channels;
      for(int channel = 0; channel < blurred.channels; ++channel){
        output.pixels[output_index + channel] = blurred.pixels[source_index + channel];
      }
    }
  }
  return output;
}

ImagePyramid build_gaussian_pyramid_omp(const FloatImage &input,
                                        const std::vector<float> &weights,
                                        int max_levels,
                                        int minimum_side)
{
  if(max_levels <= 0 || minimum_side <= 0){
    throw std::invalid_argument("pyramid level count and minimum side must be positive");
  }
  ImagePyramid pyramid;
  pyramid.push_back(input);
  while(static_cast<int>(pyramid.size()) < max_levels){
    const FloatImage &current = pyramid.back();
    const int next_width = (current.width + 1) / 2;
    const int next_height = (current.height + 1) / 2;
    if(std::min(next_width, next_height) < minimum_side){
      break;
    }
    pyramid.push_back(downsample_from_blurred_omp(blur_float_omp(current, weights)));
  }
  return pyramid;
}

ImagePyramid build_laplacian_pyramid_omp(const ImagePyramid &gaussian)
{
  if(gaussian.empty()){
    throw std::invalid_argument("Gaussian pyramid cannot be empty");
  }
  ImagePyramid laplacian(gaussian.size());
  for(std::size_t level = 0; level + 1 < gaussian.size(); ++level){
    FloatImage expanded = resize_bilinear_omp(gaussian[level + 1], gaussian[level].width, gaussian[level].height);
    laplacian[level] = gaussian[level];
    #pragma omp parallel for schedule(static)
    for(std::size_t index = 0; index < laplacian[level].pixels.size(); ++index){
      laplacian[level].pixels[index] -= expanded.pixels[index];
    }
  }
  laplacian.back() = gaussian.back();
  return laplacian;
}

FloatImage collapse_laplacian_pyramid_omp(const ImagePyramid &laplacian)
{
  if(laplacian.empty()){
    throw std::invalid_argument("Laplacian pyramid cannot be empty");
  }
  FloatImage current = laplacian.back();
  for(std::size_t level = laplacian.size() - 1; level-- > 0;){
    FloatImage expanded = resize_bilinear_omp(current, laplacian[level].width, laplacian[level].height);
    #pragma omp parallel for schedule(static)
    for(std::size_t index = 0; index < expanded.pixels.size(); ++index){
      expanded.pixels[index] += laplacian[level].pixels[index];
    }
    current = std::move(expanded);
  }
  return current;
}
