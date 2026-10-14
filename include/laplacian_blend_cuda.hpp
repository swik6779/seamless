#ifndef LAPLACIAN_BLEND_CUDA_HPP
#define LAPLACIAN_BLEND_CUDA_HPP

#include "pyramid.hpp"
#include "resize_cuda.hpp"

bool blend_images_cuda(const FloatImage &image_a,
                       const FloatImage &image_b,
                       const FloatImage &mask,
                       const std::vector<float> &weights,
                       int max_levels,
                       FloatImage &output,
                       int minimum_side = 16,
                       CudaTiming *timing = nullptr);

#endif
