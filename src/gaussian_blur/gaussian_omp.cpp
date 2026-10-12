#include "../../include/gaussian_omp.hpp"

int clamp_coordinate_omp(int coordinate, int limit)
{
  if(coordinate < 0){
    return 0;
  }
  if(coordinate >= limit){
    return limit - 1;
  }
  return coordinate;
}

stbi_uc to_byte_omp(float value)
{
  if(value < 0.0f){
    value = 0.0f;
  }
  else if(value > 255.0f){
    value = 255.0f;
  }
  return static_cast<stbi_uc>(value);
}

void gaussian_blur_omp_naive(const char *filepath, std::vector<std::vector<float>> &kernel, const char *output_filename)
{
  Image input = load_image(filepath);
  if(input.pixels == nullptr){
    return;
  }

  const int radius = static_cast<int>(kernel.size()) / 2;
  const std::size_t pixel_count = static_cast<std::size_t>(input.width) * input.height;
  std::vector<stbi_uc> output_pixels(pixel_count * input.channels);

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for(int ky = -radius; ky <= radius; ++ky){
        for(int kx = -radius; kx <= radius; ++kx){
          const int nx = clamp_coordinate_omp(x + kx, input.width);
          const int ny = clamp_coordinate_omp(y + ky, input.height);
          const int input_index = (ny * input.width + nx) * input.channels;
          const float weight = kernel[ky + radius][kx + radius];

          for(int channel = 0; channel < input.channels; ++channel){
            values[channel] += weight * input.pixels[input_index + channel];
          }
        }
      }

      const int output_index = (y * input.width + x) * input.channels;
      for(int channel = 0; channel < input.channels; ++channel){
        output_pixels[output_index + channel] = to_byte_omp(values[channel]);
      }
    }
  }

  Image output = {input.width, input.height, input.channels, output_pixels.data()};
  write_image(output_filename, &output);
  free_image(&input);
}

void gaussian_blur_omp_separable(const char *filepath, std::vector<float> &weights, const char *output_filename)
{
  Image input = load_image(filepath);
  if(input.pixels == nullptr){
    return;
  }

  const int radius = static_cast<int>(weights.size()) / 2;
  const std::size_t pixel_count = static_cast<std::size_t>(input.width) * input.height;
  const std::size_t image_size = pixel_count * input.channels;
  std::vector<float> horizontal(image_size);
  std::vector<stbi_uc> output_pixels(image_size);

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for(int kx = -radius; kx <= radius; ++kx){
        const int nx = clamp_coordinate_omp(x + kx, input.width);
        const int input_index = (y * input.width + nx) * input.channels;
        const float weight = weights[kx + radius];

        for(int channel = 0; channel < input.channels; ++channel){
          values[channel] += weight * input.pixels[input_index + channel];
        }
      }

      const int output_index = (y * input.width + x) * input.channels;
      for(int channel = 0; channel < input.channels; ++channel){
        horizontal[output_index + channel] = values[channel];
      }
    }
  }

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y = 0; y < input.height; ++y){
    for(int x = 0; x < input.width; ++x){
      float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for(int ky = -radius; ky <= radius; ++ky){
        const int ny = clamp_coordinate_omp(y + ky, input.height);
        const int input_index = (ny * input.width + x) * input.channels;
        const float weight = weights[ky + radius];

        for(int channel = 0; channel < input.channels; ++channel){
          values[channel] += weight * horizontal[input_index + channel];
        }
      }

      const int output_index = (y * input.width + x) * input.channels;
      for(int channel = 0; channel < input.channels; ++channel){
        output_pixels[output_index + channel] = to_byte_omp(values[channel]);
      }
    }
  }

  Image output = {input.width, input.height, input.channels, output_pixels.data()};
  write_image(output_filename, &output);
  free_image(&input);
}
