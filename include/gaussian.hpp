#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

double gaussian_function_value(double x, double sigma);
std::vector<std::vector<double>> create_gaussian_kernel(int k);
void gaussian_blur(const char *filepath, std::vector<std::vector<double>> &kernel);

#endif
