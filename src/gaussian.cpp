#include "../include/gaussian.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// returns the Gaussian function value for a value of x and sigma
float gaussian_function_value(float x, float sigma)
{
  if(sigma < 0.0){
    return -1;
  }

  return std::exp(-(x*x) / (2*sigma*sigma)) / (std::pow(2*std::numbers::pi, 0.5) * sigma);
}

// returns the Gaussian weights of size k
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
  float sigma = 3; //will be changed in the future, for now it is being hardcoded
  
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
  float sigma = 3; //will be changed in the future, for now it is being hardcoded
  
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

//naive implementation of gaussian blur
/*
   
    A B C
    D E F (example of 3X3 kernel) 
    G H I

    this kernel is applied onto each pixel. each pixel is manipulated using k^2 arithmetic operations
*/

void gaussian_blur_iterable_naive(const char *filepath, std::vector<std::vector<float>> &kernel){
  int height, width, channel;
  stbi_uc *decoded_vals = stbi_load(filepath, &width, &height, &channel, STBI_rgb);

  if(decoded_vals == NULL){
    fprintf(stderr, "could not load files\n");
    return;
  }

  int radius = kernel.size()/2;
  stbi_uc *smoothened_vals = (stbi_uc *)malloc(height * width * 3 * sizeof(stbi_uc));

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){

      float r = 0.0;
      float g = 0.0;
      float b = 0.0;

      for(int ky = -radius; ky <= radius; ky++){
        for(int kx = -radius; kx <= radius; kx++){

          int nx = x + kx;
          int ny = y + ky;

          // Edge replication
          if (nx < 0)
            nx = 0;
          else if (nx >= width)
            nx = width - 1;

          if (ny < 0)
            ny = 0;
          else if (ny >= height)
            ny = height - 1;

          int idx = (ny * width + nx) * 3;

          float weight =
            kernel[ky + radius][kx + radius];

          r += weight * decoded_vals[idx];
          g += weight * decoded_vals[idx + 1];
          b += weight * decoded_vals[idx + 2];
        }
      }

      int output_idx = (y * width + x) * 3;

      smoothened_vals[output_idx]     = (stbi_uc)r;
      smoothened_vals[output_idx + 1] = (stbi_uc)g;
      smoothened_vals[output_idx + 2] = (stbi_uc)b;
    }
  }
  
  if(!stbi_write_png("smoothened.png", width, height, STBI_rgb, smoothened_vals, width*3)){
    fprintf(stderr, "could not write smoothened values into PNG\n");
    return;
  }
}

void gaussian_blur_iterable_seperable(const char *filepath, std::vector<float> &weights){
  int height, width, channel;
  stbi_uc *decoded_vals = stbi_load(filepath, &width, &height, &channel, STBI_rgb);

  if(decoded_vals == NULL){
    fprintf(stderr, "could not load files\n");
    return;
  }

  int radius = weights.size()/2;
  stbi_uc *smoothened_vals = (stbi_uc *)malloc(height * width * 3 * sizeof(stbi_uc));

  for(int y=0; y<height; y++){
    for(int x=0; x<width; x++){

      float r = 0.0;
      float g = 0.0;
      float b = 0.0;
      for(kx = -radius; kx<=radius; kx++){
        int nx = kx + x;
        
      }
    }
  }
  
  
  if(!stbi_write_png("smoothened.png", width, height, STBI_rgb, smoothened_vals, width*3)){
    fprintf(stderr, "could not write smoothened values into PNG\n");
    return;
  }
}
