#ifndef PYRAMID_OMP_HPP
#define PYRAMID_OMP_HPP

#include "pyramid.hpp"

FloatImage blur_float_omp(const FloatImage &input, const std::vector<float> &weights);
FloatImage downsample_from_blurred_omp(const FloatImage &blurred);
ImagePyramid build_gaussian_pyramid_omp(const FloatImage &input,
                                        const std::vector<float> &weights,
                                        int max_levels,
                                        int minimum_side = 16);
ImagePyramid build_laplacian_pyramid_omp(const ImagePyramid &gaussian);
FloatImage collapse_laplacian_pyramid_omp(const ImagePyramid &laplacian);

#endif
