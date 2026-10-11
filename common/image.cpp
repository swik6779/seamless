#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../include/image.hpp"
#include "../include/stb_image_write.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

Image load_image(const char *filepath)
{
  Image image = {0, 0, 0, nullptr};
  image.pixels = stbi_load(filepath, &image.width, &image.height, &image.channels, 0);

  if(image.pixels == nullptr){
    fprintf(stderr, "could not load image %s: %s\n", filepath, stbi_failure_reason());
    return {0, 0, 0, nullptr};
  }

  if(image.channels < 1 || image.channels > 4){
    fprintf(stderr, "unsupported channel count %d in image %s\n", image.channels, filepath);
    stbi_image_free(image.pixels);
    return {0, 0, 0, nullptr};
  }

  return image;
}

void free_image(Image *image)
{
  if(image == nullptr){
    return;
  }

  if(image->pixels != nullptr){
    stbi_image_free(image->pixels);
  }

  image->width = 0;
  image->height = 0;
  image->channels = 0;
  image->pixels = nullptr;
}

bool write_image(const char *filepath, const Image *image)
{
  if(filepath == nullptr || image == nullptr || image->pixels == nullptr ||
     image->width <= 0 || image->height <= 0 || image->channels < 1 || image->channels > 4){
    fprintf(stderr, "invalid image data for output\n");
    return false;
  }

  const char *extension_start = strrchr(filepath, '.');
  if(extension_start == nullptr){
    fprintf(stderr, "output filename must have a supported image extension: %s\n", filepath);
    return false;
  }

  std::string extension(extension_start);
  for(char &character : extension){
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }

  const int stride = image->width * image->channels;
  int result = 0;

  if(extension == ".png"){
    result = stbi_write_png(filepath, image->width, image->height, image->channels, image->pixels, stride);
  }
  else if(extension == ".bmp"){
    result = stbi_write_bmp(filepath, image->width, image->height, image->channels, image->pixels);
  }
  else if(extension == ".tga"){
    result = stbi_write_tga(filepath, image->width, image->height, image->channels, image->pixels);
  }
  else if(extension == ".jpg" || extension == ".jpeg"){
    result = stbi_write_jpg(filepath, image->width, image->height, image->channels, image->pixels, 95);
  }
  else{
    fprintf(stderr, "unsupported output image extension: %s\n", extension.c_str());
    return false;
  }

  if(result == 0){
    fprintf(stderr, "could not write image %s\n", filepath);
    return false;
  }

  return true;
}
