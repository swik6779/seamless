#ifndef GAUSSIAN_COMMON_H
#define GAUSSIAN_COMMON_H

#include <vector>
#include <iostream>
#include <cmath>
#include <numbers>

#include "image.hpp"

float gaussian_function_value(float x, float sigma);
std::vector<float> create_gaussian_weights(int k);
std::vector<std::vector<float>> create_gaussian_kernel(int k);

#endif
