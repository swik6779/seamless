#include "../include/gaussian_common.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

float gaussian_function_value(float x, float sigma)
{
  if(sigma < 0.0){
    return -1;
  }

  return std::exp(-(x*x) / (2*sigma*sigma)) / (std::pow(2*std::numbers::pi, 0.5) * sigma);
}

std::vector<float> create_gaussian_weights(int k)
{
  if(k <= 0){
    fprintf(stderr, "cannot have  kernel_size <= 0\n");
    return {};
  }

  if(k % 2 == 0){
    fprintf(stderr, "cannot have even-valued kernel sizes\n");
    return {};
  }

  int radius = (k - 1) / 2;
  float sigma = 3;

  std::vector<float> weights(k);

  float normalization_const = 0.0;
  for(int i=0; i<k; i++){
    weights[i] = gaussian_function_value(radius - i, sigma);
    normalization_const += weights[i];
  }

  for(int i=0; i<k; i++){
    weights[i] /= normalization_const;
  }

  return weights;
}

std::vector<std::vector<float>> create_gaussian_kernel(int k)
{
  if(k <= 0){
    fprintf(stderr, "cannot have  kernel_size <= 0\n");
    return {};
  }

  if(k % 2 == 0){
    fprintf(stderr, "cannot have even-valued kernel sizes\n");
    return {};
  }

  int radius = (k - 1) / 2;
  float sigma = 3;

  std::vector<std::vector<float>> kernel(k, std::vector<float>(k, 0));

  float normalization_const = 0.0;
  for(int i=0; i<k; i++){
    for(int j=0; j<k; j++){
      kernel[i][j] = gaussian_function_value(radius - i, sigma) * gaussian_function_value(radius - j, sigma);
      normalization_const += kernel[i][j];
    }
  }

  for(int i=0; i<k; i++){
    for(int j=0; j<k; j++){
      kernel[i][j] /= normalization_const;
    }
  }

  return kernel;
}
