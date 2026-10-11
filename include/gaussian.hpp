#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "../include/gaussian_common.hpp"

void gaussian_blur_iterable_naive(const char *filepath, std::vector<std::vector<float>> &kernel, const char *output_filename);
void gaussian_blur_iterable_seperable(const char *filepath, std::vector<float> &weights, const char *output_filename);

#endif
