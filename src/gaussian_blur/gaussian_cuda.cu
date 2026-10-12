#include "../../include/gaussian_cuda.hpp"

constexpr int kMaxChannels = 4;
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

bool valid_image_dimensions(int width, int height)
{
  if(width <= 0 || height <= 0){
    fprintf(stderr, "image dimensions must be positive\n");
    return false;
  }

  return true;
}

__device__ int clamp_coordinate_cuda(int coordinate, int limit)
{
  if(coordinate < 0){
    return 0;
  }
  if(coordinate >= limit){
    return limit - 1;
  }
  return coordinate;
}

__device__ unsigned char to_byte_cuda(float value)
{
  if(value < 0.0f){
    value = 0.0f;
  }
  else if(value > 255.0f){
    value = 255.0f;
  }

  return static_cast<unsigned char>(value);
}

__global__ void gaussian_naive_kernel(const unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float values[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int ky = -radius; ky <= radius; ++ky){
    for(int kx = -radius; kx <= radius; ++kx){
      const int nx = clamp_coordinate_cuda(x + kx, width);
      const int ny = clamp_coordinate_cuda(y + ky, height);
      const int input_index = (ny * width + nx) * channels;
      const int kernel_index = (ky + radius) * kernel_size + (kx + radius);
      const float weight = kernel[kernel_index];

      for(int channel = 0; channel < channels; ++channel){
        values[channel] += weight * input[input_index + channel];
      }
    }
  }

  const int output_index = (y * width + x) * channels;
  for(int channel = 0; channel < channels; ++channel){
    output[output_index + channel] = to_byte_cuda(values[channel]);
  }
}

__global__ void gaussian_horizontal_kernel(const unsigned char *input, float *horizontal, int width, int height, int channels, const float *weights, int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float values[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int kx = -radius; kx <= radius; ++kx){
    const int nx = clamp_coordinate_cuda(x + kx, width);
    const int input_index = (y * width + nx) * channels;
    const float weight = weights[kx + radius];

    for(int channel = 0; channel < channels; ++channel){
      values[channel] += weight * input[input_index + channel];
    }
  }

  const int output_index = (y * width + x) * channels;
  for(int channel = 0; channel < channels; ++channel){
    horizontal[output_index + channel] = values[channel];
  }
}

__global__ void gaussian_vertical_kernel(const float *horizontal, unsigned char *output, int width, int height, int channels, const float *weights, int kernel_size)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height){
    return;
  }

  const int radius = kernel_size / 2;
  float values[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int ky = -radius; ky <= radius; ++ky){
    const int ny = clamp_coordinate_cuda(y + ky, height);
    const int input_index = (ny * width + x) * channels;
    const float weight = weights[ky + radius];

    for(int channel = 0; channel < channels; ++channel){
      values[channel] += weight * horizontal[input_index + channel];
    }
  }

  const int output_index = (y * width + x) * channels;
  for(int channel = 0; channel < channels; ++channel){
    output[output_index + channel] = to_byte_cuda(values[channel]);
  }
}

bool copy_to_device(void *destination, const void *source, std::size_t bytes, const char *operation)
{
  return report_cuda_error(operation, cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

bool copy_from_device(void *destination, const void *source, std::size_t bytes, const char *operation)
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

void free_device_buffers(unsigned char *device_input, unsigned char *device_output, float *device_kernel)
{
  cudaFree(device_kernel);
  cudaFree(device_output);
  cudaFree(device_input);
}

void free_device_buffers(unsigned char *device_input, unsigned char *device_output, float *device_horizontal, float *device_weights)
{
  cudaFree(device_weights);
  cudaFree(device_horizontal);
  cudaFree(device_output);
  cudaFree(device_input);
}

void gaussian_blur_cuda_naive(const char *filepath, const std::vector<std::vector<float>> &kernel, const char *output_filename)
{
  Image input = load_image(filepath);
  if(input.pixels == nullptr){
    return;
  }

  if(!valid_image_dimensions(input.width, input.height)){
    free_image(&input);
    return;
  }

  const std::size_t pixel_count = static_cast<std::size_t>(input.width) * input.height;
  const std::size_t channel_count = pixel_count * input.channels;
  const std::size_t image_bytes = channel_count * sizeof(stbi_uc);
  const std::size_t kernel_element_count = kernel.size() * kernel.size();
  const std::size_t kernel_bytes = kernel_element_count * sizeof(float);

  std::vector<float> flat_kernel;
  flat_kernel.reserve(kernel_element_count);
  for(const std::vector<float> &row : kernel){
    flat_kernel.insert(flat_kernel.end(), row.begin(), row.end());
  }

  unsigned char *device_input = nullptr;
  unsigned char *device_output = nullptr;
  float *device_kernel = nullptr;
  std::vector<stbi_uc> output_pixels(channel_count);

  if(!report_cuda_error("allocating input image", cudaMalloc(reinterpret_cast<void **>(&device_input), image_bytes)) ||
     !report_cuda_error("allocating output image", cudaMalloc(reinterpret_cast<void **>(&device_output), image_bytes)) ||
     !report_cuda_error("allocating naive kernel", cudaMalloc(reinterpret_cast<void **>(&device_kernel), kernel_bytes)) ||
     !copy_to_device(device_input, input.pixels, image_bytes, "copying input image") ||
     !copy_to_device(device_kernel, flat_kernel.data(), kernel_bytes, "copying naive kernel")){
    free_device_buffers(device_input, device_output, device_kernel);
    free_image(&input);
    return;
  }

  const dim3 block(kBlockWidth, kBlockHeight);
  const dim3 grid((static_cast<unsigned int>(input.width) + block.x - 1) / block.x, (static_cast<unsigned int>(input.height) + block.y - 1) / block.y);
  gaussian_naive_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height, input.channels, device_kernel, static_cast<int>(kernel.size()));

  if(!finish_kernel("naive Gaussian kernel") ||
     !copy_from_device(output_pixels.data(), device_output, image_bytes, "copying naive output")){
    free_device_buffers(device_input, device_output, device_kernel);
    free_image(&input);
    return;
  }

  Image output = {input.width, input.height, input.channels, output_pixels.data()};
  write_image(output_filename, &output);
  free_device_buffers(device_input, device_output, device_kernel);
  free_image(&input);
}

void gaussian_blur_cuda_separable(const char *filepath, const std::vector<float> &weights, const char *output_filename)
{
  Image input = load_image(filepath);
  if(input.pixels == nullptr){
    return;
  }

  if(!valid_image_dimensions(input.width, input.height)){
    free_image(&input);
    return;
  }

  const std::size_t pixel_count = static_cast<std::size_t>(input.width) * input.height;
  const std::size_t channel_count = pixel_count * input.channels;
  const std::size_t image_bytes = channel_count * sizeof(stbi_uc);
  const std::size_t weights_bytes = weights.size() * sizeof(float);

  unsigned char *device_input = nullptr;
  unsigned char *device_output = nullptr;
  float *device_horizontal = nullptr;
  float *device_weights = nullptr;
  std::vector<stbi_uc> output_pixels(channel_count);

  if(!report_cuda_error("allocating input image", cudaMalloc(reinterpret_cast<void **>(&device_input), image_bytes)) ||
     !report_cuda_error("allocating output image", cudaMalloc(reinterpret_cast<void **>(&device_output), image_bytes)) ||
     !report_cuda_error("allocating horizontal intermediate image", cudaMalloc(reinterpret_cast<void **>(&device_horizontal), channel_count * sizeof(float))) ||
     !report_cuda_error("allocating separable weights", cudaMalloc(reinterpret_cast<void **>(&device_weights), weights_bytes)) ||
     !copy_to_device(device_input, input.pixels, image_bytes, "copying input image") ||
     !copy_to_device(device_weights, weights.data(), weights_bytes, "copying separable weights")){
    free_device_buffers(device_input, device_output, device_horizontal, device_weights);
    free_image(&input);
    return;
  }

  const dim3 block(kBlockWidth, kBlockHeight);
  const dim3 grid((static_cast<unsigned int>(input.width) + block.x - 1) / block.x, (static_cast<unsigned int>(input.height) + block.y - 1) / block.y);
  gaussian_horizontal_kernel<<<grid, block>>>(device_input, device_horizontal, input.width, input.height, input.channels, device_weights, static_cast<int>(weights.size()));

  if(!finish_kernel("horizontal Gaussian kernel")){
    free_device_buffers(device_input, device_output, device_horizontal, device_weights);
    free_image(&input);
    return;
  }

  gaussian_vertical_kernel<<<grid, block>>>(device_horizontal, device_output, input.width, input.height, input.channels, device_weights, static_cast<int>(weights.size()));

  if(!finish_kernel("vertical Gaussian kernel") ||
     !copy_from_device(output_pixels.data(), device_output, image_bytes, "copying separable output")){
    free_device_buffers(device_input, device_output, device_horizontal, device_weights);
    free_image(&input);
    return;
  }

  Image output = {input.width, input.height, input.channels, output_pixels.data()};
  write_image(output_filename, &output);
  free_device_buffers(device_input, device_output, device_horizontal, device_weights);
  free_image(&input);
}
