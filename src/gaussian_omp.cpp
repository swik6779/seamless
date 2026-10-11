#include "../include/gaussian_omp.hpp"
#include "../include/gaussian_common.hpp"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <omp.h>

// OpenMP parallelized naive 2D convolution
// Each pixel output is independent, so we collapse the outer two loops across all threads
void gaussian_blur_omp_naive(const char *filepath, std::vector<std::vector<float>> &kernel, const char *output_filename)
{
  int height, width, channel;
  stbi_uc *decoded_vals = stbi_load(filepath, &width, &height, &channel, STBI_rgb);

  if(decoded_vals == NULL){
    fprintf(stderr, "could not load file\n");
    return;
  }

  int radius = kernel.size() / 2;
  stbi_uc *smoothened_vals = (stbi_uc *)malloc(height * width * 3 * sizeof(stbi_uc));

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){

      float r = 0.0f;
      float g = 0.0f;
      float b = 0.0f;

      for(int ky = -radius; ky <= radius; ky++){
        for(int kx = -radius; kx <= radius; kx++){

          int nx = x + kx;
          int ny = y + ky;

          // Edge replication
          if(nx < 0) nx = 0;
          else if(nx >= width)  nx = width - 1;
          if(ny < 0) ny = 0;
          else if(ny >= height) ny = height - 1;

          int idx = (ny * width + nx) * 3;
          float weight = kernel[ky + radius][kx + radius];

          r += weight * decoded_vals[idx];
          g += weight * decoded_vals[idx + 1];
          b += weight * decoded_vals[idx + 2];
        }
      }

      int output_idx = (y * width + x) * 3;
      smoothened_vals[output_idx] = (stbi_uc)r;
      smoothened_vals[output_idx + 1] = (stbi_uc)g;
      smoothened_vals[output_idx + 2] = (stbi_uc)b;
    }
  }

  if(!stbi_write_png(output_filename, width, height, STBI_rgb, smoothened_vals, width * 3)){
    fprintf(stderr, "could not write smoothened_omp_naive.png\n");
  }

  stbi_image_free(decoded_vals);
  free(smoothened_vals);
}

// OpenMP parallelized separable convolution
// Horizontal and vertical passes are each parallelized independently
void gaussian_blur_omp_separable(const char *filepath, std::vector<float> &weights, const char *output_filename)
{
  int height, width, channel;
  stbi_uc *decoded_vals = stbi_load(filepath, &width, &height, &channel, STBI_rgb);

  if(decoded_vals == NULL){
    fprintf(stderr, "could not load file\n");
    return;
  }

  int radius = weights.size() / 2;
  size_t image_size = (size_t)width * height * 3;

  std::vector<float> horizontal(image_size);
  stbi_uc *smoothened_vals = (stbi_uc *)malloc(image_size * sizeof(stbi_uc));

  //  Horizontal pass 
  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){

      float r = 0.0f, g = 0.0f, b = 0.0f;

      for(int kx = -radius; kx <= radius; kx++){
        int nx = x + kx;
        if(nx < 0)        nx = 0;
        else if(nx >= width) nx = width - 1;

        int input_idx = (y * width + nx) * 3;
        float weight  = weights[kx + radius];

        r += weight * decoded_vals[input_idx];
        g += weight * decoded_vals[input_idx + 1];
        b += weight * decoded_vals[input_idx + 2];
      }

      int output_idx = (y * width + x) * 3;
      horizontal[output_idx] = r;
      horizontal[output_idx + 1] = g;
      horizontal[output_idx + 2] = b;
    }
  }

  // Vertical pass 
  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){

      float r = 0.0f, g = 0.0f, b = 0.0f;

      for(int ky = -radius; ky <= radius; ky++){
        int ny = y + ky;
        if(ny < 0) ny = 0;
        else if(ny >= height) ny = height - 1;

        int input_idx = (ny * width + x) * 3;
        float weight  = weights[ky + radius];

        r += weight * horizontal[input_idx];
        g += weight * horizontal[input_idx + 1];
        b += weight * horizontal[input_idx + 2];
      }

      int output_idx = (y * width + x) * 3;
      smoothened_vals[output_idx] = (stbi_uc)(r > 255.0f ? 255.0f : r);
      smoothened_vals[output_idx + 1] = (stbi_uc)(g > 255.0f ? 255.0 : g);
      smoothened_vals[output_idx + 2] = (stbi_uc)(b > 255.0f ? 255.0f : b);
    }
  }

  if(!stbi_write_png(output_filename, width, height, STBI_rgb, smoothened_vals, width * 3)){
    fprintf(stderr, "could not write smoothened_omp_separable.png\n");
  }

  stbi_image_free(decoded_vals);
  free(smoothened_vals);
}
