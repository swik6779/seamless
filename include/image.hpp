#ifndef IMAGE_H
#define IMAGE_H

#include "../include/stb_image.h"

struct Image {
  int width;
  int height;
  int channels;
  stbi_uc *pixels;
};

Image load_image(const char *filepath);
void free_image(Image *image);
bool write_image(const char *filepath, const Image *image);

#endif
