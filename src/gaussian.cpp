#include "../include/gaussian.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// returns the Gaussian function value for a value of x and sigma
double gaussian_function_value(double x, double sigma)
{
  if(sigma < 0.0){
    return -1;
  }

  return std::exp(-(x*x) / (2*sigma*sigma)) / (std::pow(2*std::numbers::pi, 0.5) * sigma);
}

// returns the Gaussian kernel of size k X k
std::vector<std::vector<double>> create_gaussian_kernel(int k)
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
  double sigma = 3; //will be changed in the future, for now it is being hardcoded
  
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

//gaussian blur implementation
void gaussian_blur(const char *filepath, std::vector<std::vector<double>> &kernel)
{
  int height, width, channel;
  stbi_uc *decoded_vals = stbi_load(filepath, &width, &height, &channel, STBI_rgb);

  if(decoded_vals == NULL){
    fprintf(stderr, "could not load files\n");
    return;
  }
  
  int radius = kernel.size()/2;
  stbi_uc *smoothened_vals = (stbi_uc *)malloc(height * width * 3 * sizeof(stbi_uc));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

      double r = 0.0;
      double g = 0.0;
      double b = 0.0;

      for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {

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

          double weight =
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
