#include "../include/gaussian.hpp"

double gaussian_function_value(double x, double sigma){
  if(sigma < 0.0){
    return -1;
  }

  return std::exp(-(x*x) / (2*sigma*sigma)) / (std::pow(2*std::numbers::pi, 0.5) * sigma);
}

std::vector<std::vector<double>> create_gaussian_kernel(int k){
  if(k <= 0){
    return {};
  }

  if(k % 2 == 0){
    return {};
  }

  int radius = (k - 1) / 2;
  double sigma = 1; //will be changed in the future, for now it is being hardcoded
  
  std::vector<std::vector<double>> kernel(k, std::vector<double>(k, 0));

  double normalization_const = 0.0;
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
