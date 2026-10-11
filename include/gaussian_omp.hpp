#ifndef GAUSSIAN_OMP_H
#define GAUSSIAN_OMP_H

#include "../include/gaussian_common.hpp"

void gaussian_blur_omp_naive(const char *filepath, std::vector<std::vector<float>> &kernel, const char *output_filename);
void gaussian_blur_omp_separable(const char *filepath, std::vector<float> &weights, const char *output_filename);

#endif
