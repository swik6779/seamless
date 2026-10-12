#include "../include/pyramid.hpp"

#include <omp.h>
#include <stdexcept>

namespace {

ImagePyramid blend_laplacian_omp(const ImagePyramid &laplacian_a,
                                 const ImagePyramid &laplacian_b,
                                 const ImagePyramid &mask)
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
    #pragma omp parallel for collapse(2) schedule(static)
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
  }
  return blended;
}

}

FloatImage blend_images_omp(const FloatImage &image_a,
                            const FloatImage &image_b,
                            const FloatImage &mask,
                            const std::vector<float> &weights,
                            int max_levels,
                            int minimum_side)
{
  if(image_a.width != image_b.width || image_a.height != image_b.height || image_a.channels != image_b.channels ||
     mask.width != image_a.width || mask.height != image_a.height || mask.channels != 1){
    throw std::invalid_argument("blend inputs have incompatible dimensions or channels");
  }

  const ImagePyramid gaussian_a = build_gaussian_pyramid(image_a, weights, max_levels, minimum_side);
  const ImagePyramid gaussian_b = build_gaussian_pyramid(image_b, weights, max_levels, minimum_side);
  const ImagePyramid gaussian_mask = build_gaussian_pyramid(mask, weights, max_levels, minimum_side);
  return collapse_laplacian_pyramid(blend_laplacian_omp(build_laplacian_pyramid(gaussian_a),
                                                        build_laplacian_pyramid(gaussian_b),
                                                        gaussian_mask));
}
