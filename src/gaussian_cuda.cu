#include "../include/gaussian_cuda.hpp"

#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr int kChannels = 3;
constexpr int kBlockWidth = 16;
constexpr int kBlockHeight = 16;

bool report_cuda_error(const char *operation, cudaError_t error)
{
  if(error == cudaSuccess){
    return true;
  }

  fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
  return false;
}

template <typename T>
class device_buffer {
public:
  device_buffer() = default;
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;

  ~device_buffer()
  {
    if(pointer_ != nullptr){
      cudaFree(pointer_);
    }
  }

  bool allocate(std::size_t element_count, const char *name)
  {
    if(element_count == 0){
      fprintf(stderr, "cannot allocate an empty CUDA buffer for %s\n", name);
      return false;
    }

    return report_cuda_error(
        name,
        cudaMalloc(reinterpret_cast<void **>(&pointer_), element_count * sizeof(T)));
  }

  T *get() const
  {
    return pointer_;
  }

private:
  T *pointer_ = nullptr;
};

bool valid_kernel(const std::vector<std::vector<float>> &kernel)
{
  const std::size_t size = kernel.size();
  if(size == 0 || size % 2 == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max())){
    fprintf(stderr, "CUDA naive convolution requires a non-empty odd-sized kernel\n");
    return false;
  }

  for(const std::vector<float> &row : kernel){
    if(row.size() != size){
      fprintf(stderr, "CUDA naive convolution requires a square kernel\n");
      return false;
    }
  }

  return true;
}

bool valid_weights(const std::vector<float> &weights)
{
  if(weights.empty() || weights.size() % 2 == 0 ||
     weights.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())){
    fprintf(stderr, "CUDA separable convolution requires non-empty odd-sized weights\n");
    return false;
  }

  return true;
}

bool valid_image_dimensions(int width, int height)
{
  if(width <= 0 || height <= 0){
    fprintf(stderr, "image dimensions must be positive\n");
    return false;
  }

  return true;
}

__device__ int clamp_coordinate(int coordinate, int limit)
{
  if(coordinate < 0){
    return 0;
  }
  if(coordinate >= limit){
    return limit - 1;
  }
  return coordinate;
}

__device__ unsigned char to_byte(float value)
{
  if(value < 0.0f){
    value = 0.0f;
  }
  else if(value > 255.0f){
    value = 255.0f;
  }

  return static_cast<unsigned char>(value);
}

__global__ void gaussian_naive_kernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height,
    const float *kernel,
    int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float channels[kChannels] = {0.0f, 0.0f, 0.0f};

  for(int ky = -radius; ky <= radius; ++ky){
    for(int kx = -radius; kx <= radius; ++kx){
      const int nx = clamp_coordinate(x + kx, width);
      const int ny = clamp_coordinate(y + ky, height);
      const int input_index = (ny * width + nx) * kChannels;
      const int kernel_index = (ky + radius) * kernel_size + (kx + radius);
      const float weight = kernel[kernel_index];

      for(int channel = 0; channel < kChannels; ++channel){
        channels[channel] += weight * input[input_index + channel];
      }
    }
  }

  const int output_index = (y * width + x) * kChannels;
  for(int channel = 0; channel < kChannels; ++channel){
    output[output_index + channel] = to_byte(channels[channel]);
  }
}

__global__ void gaussian_horizontal_kernel(
    const unsigned char *input,
    float *horizontal,
    int width,
    int height,
    const float *weights,
    int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float channels[kChannels] = {0.0f, 0.0f, 0.0f};

  for(int kx = -radius; kx <= radius; ++kx){
    const int nx = clamp_coordinate(x + kx, width);
    const int input_index = (y * width + nx) * kChannels;
    const float weight = weights[kx + radius];

    for(int channel = 0; channel < kChannels; ++channel){
      channels[channel] += weight * input[input_index + channel];
    }
  }

  const int output_index = (y * width + x) * kChannels;
  for(int channel = 0; channel < kChannels; ++channel){
    horizontal[output_index + channel] = channels[channel];
  }
}

__global__ void gaussian_vertical_kernel(
    const float *horizontal,
    unsigned char *output,
    int width,
    int height,
    const float *weights,
    int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float channels[kChannels] = {0.0f, 0.0f, 0.0f};

  for(int ky = -radius; ky <= radius; ++ky){
    const int ny = clamp_coordinate(y + ky, height);
    const int input_index = (ny * width + x) * kChannels;
    const float weight = weights[ky + radius];

    for(int channel = 0; channel < kChannels; ++channel){
      channels[channel] += weight * horizontal[input_index + channel];
    }
  }

  const int output_index = (y * width + x) * kChannels;
  for(int channel = 0; channel < kChannels; ++channel){
    output[output_index + channel] = to_byte(channels[channel]);
  }
}

bool copy_to_device(
    void *destination,
    const void *source,
    std::size_t bytes,
    const char *operation)
{
  return report_cuda_error(operation, cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

bool copy_from_device(
    void *destination,
    const void *source,
    std::size_t bytes,
    const char *operation)
{
  return report_cuda_error(operation, cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

bool finish_kernel(const char *kernel_name)
{
  if(!report_cuda_error(kernel_name, cudaGetLastError())){
    return false;
  }

  return report_cuda_error("CUDA kernel synchronization", cudaDeviceSynchronize());
}

bool write_png(
    const char *output_filename,
    int width,
    int height,
    const unsigned char *pixels)
{
  if(stbi_write_png(output_filename, width, height, kChannels, pixels, width * kChannels) == 0){
    fprintf(stderr, "could not write CUDA blur output to %s\n", output_filename);
    return false;
  }

  return true;
}

} // namespace

void gaussian_blur_cuda_naive(
    const char *filepath,
    const std::vector<std::vector<float>> &kernel,
    const char *output_filename)
{
  if(!valid_kernel(kernel)){
    return;
  }

  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  stbi_uc *input_pixels = stbi_load(filepath, &width, &height, &channels_in_file, STBI_rgb);
  if(input_pixels == nullptr){
    fprintf(stderr, "could not load file for CUDA naive blur\n");
    return;
  }

  if(!valid_image_dimensions(width, height)){
    stbi_image_free(input_pixels);
    return;
  }

  const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
  const std::size_t image_bytes = pixel_count * kChannels * sizeof(stbi_uc);
  const std::size_t kernel_element_count = kernel.size() * kernel.size();
  const std::size_t kernel_bytes = kernel_element_count * sizeof(float);

  std::vector<float> flat_kernel;
  flat_kernel.reserve(kernel_element_count);
  for(const std::vector<float> &row : kernel){
    flat_kernel.insert(flat_kernel.end(), row.begin(), row.end());
  }

  device_buffer<stbi_uc> device_input;
  device_buffer<stbi_uc> device_output;
  device_buffer<float> device_kernel;
  std::vector<stbi_uc> output_pixels(pixel_count * kChannels);

  if(!device_input.allocate(pixel_count * kChannels, "input image") ||
     !device_output.allocate(pixel_count * kChannels, "output image") ||
     !device_kernel.allocate(kernel_element_count, "naive kernel") ||
     !copy_to_device(device_input.get(), input_pixels, image_bytes, "copying input image") ||
     !copy_to_device(device_kernel.get(), flat_kernel.data(), kernel_bytes, "copying naive kernel")){
    stbi_image_free(input_pixels);
    return;
  }

  const dim3 block(kBlockWidth, kBlockHeight);
  const dim3 grid(
      (static_cast<unsigned int>(width) + block.x - 1) / block.x,
      (static_cast<unsigned int>(height) + block.y - 1) / block.y);

  gaussian_naive_kernel<<<grid, block>>>(
      device_input.get(),
      device_output.get(),
      width,
      height,
      device_kernel.get(),
      static_cast<int>(kernel.size()));

  if(!finish_kernel("naive Gaussian kernel") ||
     !copy_from_device(output_pixels.data(), device_output.get(), image_bytes, "copying naive output")){
    stbi_image_free(input_pixels);
    return;
  }

  write_png(output_filename, width, height, output_pixels.data());
  stbi_image_free(input_pixels);
}

void gaussian_blur_cuda_separable(
    const char *filepath,
    const std::vector<float> &weights,
    const char *output_filename)
{
  if(!valid_weights(weights)){
    return;
  }

  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  stbi_uc *input_pixels = stbi_load(filepath, &width, &height, &channels_in_file, STBI_rgb);
  if(input_pixels == nullptr){
    fprintf(stderr, "could not load file for CUDA separable blur\n");
    return;
  }

  if(!valid_image_dimensions(width, height)){
    stbi_image_free(input_pixels);
    return;
  }

  const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
  const std::size_t channel_count = pixel_count * kChannels;
  const std::size_t image_bytes = channel_count * sizeof(stbi_uc);
  const std::size_t weights_bytes = weights.size() * sizeof(float);

  device_buffer<stbi_uc> device_input;
  device_buffer<stbi_uc> device_output;
  device_buffer<float> device_horizontal;
  device_buffer<float> device_weights;
  std::vector<stbi_uc> output_pixels(channel_count);

  if(!device_input.allocate(channel_count, "input image") ||
     !device_output.allocate(channel_count, "output image") ||
     !device_horizontal.allocate(channel_count, "horizontal intermediate image") ||
     !device_weights.allocate(weights.size(), "separable weights") ||
     !copy_to_device(device_input.get(), input_pixels, image_bytes, "copying input image") ||
     !copy_to_device(device_weights.get(), weights.data(), weights_bytes, "copying separable weights")){
    stbi_image_free(input_pixels);
    return;
  }

  const dim3 block(kBlockWidth, kBlockHeight);
  const dim3 grid(
      (static_cast<unsigned int>(width) + block.x - 1) / block.x,
      (static_cast<unsigned int>(height) + block.y - 1) / block.y);

  gaussian_horizontal_kernel<<<grid, block>>>(
      device_input.get(),
      device_horizontal.get(),
      width,
      height,
      device_weights.get(),
      static_cast<int>(weights.size()));

  if(!finish_kernel("horizontal Gaussian kernel")){
    stbi_image_free(input_pixels);
    return;
  }

  gaussian_vertical_kernel<<<grid, block>>>(
      device_horizontal.get(),
      device_output.get(),
      width,
      height,
      device_weights.get(),
      static_cast<int>(weights.size()));

  if(!finish_kernel("vertical Gaussian kernel") ||
     !copy_from_device(output_pixels.data(), device_output.get(), image_bytes, "copying separable output")){
    stbi_image_free(input_pixels);
    return;
  }

  write_png(output_filename, width, height, output_pixels.data());
  stbi_image_free(input_pixels);
}
