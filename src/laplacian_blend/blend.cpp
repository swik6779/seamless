#include "../../include/pyramid.hpp"

#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <utility>

namespace {

void validate_blend_inputs(const FloatImage &image_a, const FloatImage &image_b, const FloatImage &mask)
{
  if(image_a.width != image_b.width || image_a.height != image_b.height || image_a.channels != image_b.channels){
    throw std::invalid_argument("input images must have matching dimensions and channels");
  }
  if(mask.width != image_a.width || mask.height != image_a.height || mask.channels != 1){
    throw std::invalid_argument("mask must match the input dimensions and have one channel");
  }
}

ImagePyramid build_gaussian_timed(const FloatImage &input,
                                  const std::vector<float> &weights,
                                  int max_levels,
                                  int minimum_side,
                                  StageTiming *timing)
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
    const double blur_start = omp_get_wtime();
    const FloatImage blurred = blur_float_cpu(current, weights);
    if(timing != nullptr){ timing->blur_ms += (omp_get_wtime() - blur_start) * 1000.0; }
    const double downsample_start = omp_get_wtime();
    pyramid.push_back(downsample_from_blurred(blurred));
    if(timing != nullptr){ timing->downsample_ms += (omp_get_wtime() - downsample_start) * 1000.0; }
  }
  return pyramid;
}

ImagePyramid build_laplacian_timed(const ImagePyramid &gaussian, StageTiming *timing)
{
  if(gaussian.empty()){
    throw std::invalid_argument("Gaussian pyramid cannot be empty");
  }
  ImagePyramid laplacian(gaussian.size());
  for(std::size_t level = 0; level + 1 < gaussian.size(); ++level){
    const double expand_start = omp_get_wtime();
    const FloatImage expanded = expand_bilinear(gaussian[level + 1], gaussian[level].width, gaussian[level].height);
    if(timing != nullptr){ timing->expand_ms += (omp_get_wtime() - expand_start) * 1000.0; }
    laplacian[level] = gaussian[level];
    const double subtract_start = omp_get_wtime();
    for(std::size_t index = 0; index < laplacian[level].pixels.size(); ++index){
      laplacian[level].pixels[index] -= expanded.pixels[index];
    }
    if(timing != nullptr){ timing->subtract_ms += (omp_get_wtime() - subtract_start) * 1000.0; }
  }
  laplacian.back() = gaussian.back();
  return laplacian;
}

ImagePyramid blend_laplacian_cpu(const ImagePyramid &laplacian_a,
                                 const ImagePyramid &laplacian_b,
                                 const ImagePyramid &mask,
                                 StageTiming *timing)
{
  if(laplacian_a.size() != laplacian_b.size() || laplacian_a.size() != mask.size() || laplacian_a.empty()){
    throw std::invalid_argument("pyramids must have matching non-zero level counts");
  }

  ImagePyramid blended(laplacian_a.size());
  for(std::size_t level = 0; level < blended.size(); ++level){
    if(laplacian_a[level].width != laplacian_b[level].width ||
       laplacian_a[level].height != laplacian_b[level].height ||
       laplacian_a[level].channels != laplacian_b[level].channels ||
       mask[level].width != laplacian_a[level].width ||
       mask[level].height != laplacian_a[level].height ||
       mask[level].channels != 1){
      throw std::invalid_argument("pyramid levels do not match");
    }

    blended[level] = laplacian_a[level];
    const double blend_start = omp_get_wtime();
    for(int y = 0; y < blended[level].height; ++y){
      for(int x = 0; x < blended[level].width; ++x){
        const std::size_t pixel_index = static_cast<std::size_t>(y) * blended[level].width + x;
        const float mask_value = mask[level].pixels[pixel_index];
        const std::size_t image_index = pixel_index * blended[level].channels;
        for(int channel = 0; channel < blended[level].channels; ++channel){
          blended[level].pixels[image_index + channel] =
              mask_value * laplacian_a[level].pixels[image_index + channel] +
              (1.0f - mask_value) * laplacian_b[level].pixels[image_index + channel];
        }
      }
    }
    if(timing != nullptr){ timing->blend_ms += (omp_get_wtime() - blend_start) * 1000.0; }
  }
  return blended;
}

FloatImage collapse_timed(const ImagePyramid &laplacian, StageTiming *timing)
{
  if(laplacian.empty()){
    throw std::invalid_argument("Laplacian pyramid cannot be empty");
  }
  FloatImage current = laplacian.back();
  for(std::size_t level = laplacian.size() - 1; level-- > 0;){
    const double collapse_start = omp_get_wtime();
    FloatImage expanded = expand_bilinear(current, laplacian[level].width, laplacian[level].height);
    for(std::size_t index = 0; index < expanded.pixels.size(); ++index){
      expanded.pixels[index] += laplacian[level].pixels[index];
    }
    if(timing != nullptr){ timing->collapse_ms += (omp_get_wtime() - collapse_start) * 1000.0; }
    current = std::move(expanded);
  }
  return current;
}

}

FloatImage blend_images_cpu(const FloatImage &image_a,
                            const FloatImage &image_b,
                            const FloatImage &mask,
                            const std::vector<float> &weights,
                            int max_levels,
                            int minimum_side,
                            StageTiming *timing)
{
  validate_blend_inputs(image_a, image_b, mask);
  const ImagePyramid gaussian_a = build_gaussian_timed(image_a, weights, max_levels, minimum_side, timing);
  const ImagePyramid gaussian_b = build_gaussian_timed(image_b, weights, max_levels, minimum_side, timing);
  const ImagePyramid gaussian_mask = build_gaussian_timed(mask, weights, max_levels, minimum_side, timing);
  const ImagePyramid laplacian_a = build_laplacian_timed(gaussian_a, timing);
  const ImagePyramid laplacian_b = build_laplacian_timed(gaussian_b, timing);
  return collapse_timed(blend_laplacian_cpu(laplacian_a, laplacian_b, gaussian_mask, timing), timing);
}
