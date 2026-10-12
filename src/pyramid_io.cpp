#include "../include/pyramid_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

FloatImage image_to_float(const Image &image)
{
  if(image.pixels == nullptr || image.width <= 0 || image.height <= 0 || image.channels <= 0){
    throw std::invalid_argument("invalid input image");
  }

  FloatImage output{image.width, image.height, image.channels,
                    std::vector<float>(static_cast<std::size_t>(image.width) * image.height * image.channels)};
  for(std::size_t index = 0; index < output.pixels.size(); ++index){
    output.pixels[index] = static_cast<float>(image.pixels[index]) / 255.0f;
  }
  return output;
}

FloatImage mask_to_float(const Image &image)
{
  if(image.pixels == nullptr || image.width <= 0 || image.height <= 0 || image.channels <= 0){
    throw std::invalid_argument("invalid mask image");
  }

  FloatImage output{image.width, image.height, 1,
                    std::vector<float>(static_cast<std::size_t>(image.width) * image.height)};
  for(std::size_t pixel = 0; pixel < output.pixels.size(); ++pixel){
    float value = 0.0f;
    const int color_channels = std::min(image.channels, 3);
    for(int channel = 0; channel < color_channels; ++channel){
      value += static_cast<float>(image.pixels[pixel * image.channels + channel]) / 255.0f;
    }
    output.pixels[pixel] = value / color_channels;
  }
  return output;
}

bool write_float_image(const char *filepath, const FloatImage &image)
{
  if(filepath == nullptr || image.width <= 0 || image.height <= 0 || image.channels < 1 || image.channels > 4 ||
     image.pixels.size() != image.size()){
    fprintf(stderr, "invalid float image for output\n");
    return false;
  }

  std::vector<stbi_uc> pixels(image.pixels.size());
  for(std::size_t index = 0; index < pixels.size(); ++index){
    const float clamped = std::clamp(image.pixels[index], 0.0f, 1.0f);
    pixels[index] = static_cast<stbi_uc>(std::lround(clamped * 255.0f));
  }

  Image output{image.width, image.height, image.channels, pixels.data()};
  return write_image(filepath, &output);
}
