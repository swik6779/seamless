#ifndef RESIZE_HPP
#define RESIZE_HPP

#include "pyramid.hpp"

FloatImage resize_bilinear(const FloatImage &input, int target_width, int target_height);
FloatImage resize_bilinear_omp(const FloatImage &input, int target_width, int target_height);

#endif
