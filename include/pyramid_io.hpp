#ifndef PYRAMID_IO_HPP
#define PYRAMID_IO_HPP

#include "image.hpp"
#include "pyramid.hpp"

FloatImage image_to_float(const Image &image);
FloatImage mask_to_float(const Image &image);
bool write_float_image(const char *filepath, const FloatImage &image);

#endif
