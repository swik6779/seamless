#ifndef GAUSSIAN_CUDA_H
#define GAUSSIAN_CUDA_H

#include "../include/gaussian_common.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <vector>

void gaussian_blur_cuda_naive(const char *filepath, const std::vector<std::vector<float>> &kernel, const char *output_filename);

void gaussian_blur_cuda_separable(const char *filepath, const std::vector<float> &weights, const char *output_filename);

#endif
