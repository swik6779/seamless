#ifndef GAUSSIAN_OMP_H
#define GAUSSIAN_OMP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

void gaussian_blur_omp_naive(const char *filepath, std::vector<std::vector<float>> &kernel);
void gaussian_blur_omp_separable(const char *filepath, std::vector<float> &weights);

#endif
