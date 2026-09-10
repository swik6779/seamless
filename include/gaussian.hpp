#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

float gaussian_function_value(float x, float sigma);
std::vector<float> create_gaussian_weights(int k);
std::vector<std::vector<float>> create_gaussian_kernel(int k);

void gaussian_blur_iterable_naive(const char *filepath, std::vector<std::vector<float>> &kernel);
void gaussian_blur_iterabale_seperable(const char *filepath, std::vector<float> &weights);

#endif
