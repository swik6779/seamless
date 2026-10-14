#include "../../include/laplacian_blend_cuda.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#define CUDA_FAIL(call) do { \
  const cudaError_t error__ = (call); \
  if(error__ != cudaSuccess){ \
    fprintf(stderr, "CUDA error at %s:%d during %s: %s\\n", __FILE__, __LINE__, #call, cudaGetErrorString(error__)); \
    cleanup(); \
    return false; \
  } \
} while(false)

#define CUDA_CLEANUP(call) do { \
  const cudaError_t error__ = (call); \
  if(error__ != cudaSuccess){ \
    fprintf(stderr, "CUDA cleanup error at %s:%d during %s: %s\\n", __FILE__, __LINE__, #call, cudaGetErrorString(error__)); \
  } \
} while(false)

namespace {

struct DeviceLevel {
  int width = 0;
  int height = 0;
  float *gaussian_a = nullptr;
  float *gaussian_b = nullptr;
  float *gaussian_mask = nullptr;
  float *laplacian_a = nullptr;
  float *laplacian_b = nullptr;
  float *blended = nullptr;
  float *expanded = nullptr;
};

__device__ int clamp_index(int value, int limit)
{
  return max(0, min(value, limit - 1));
}

__global__ void gaussian_downsample_kernel(const float *input, float *output,
                                           int input_width, int input_height, int channels,
                                           int output_width, int output_height,
                                           const float *weights, int radius)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= output_width || y >= output_height){
    return;
  }

  const std::size_t output_index = (static_cast<std::size_t>(y) * output_width + x) * channels;
  for(int channel = 0; channel < channels; ++channel){
    float value = 0.0f;
    for(int ky = -radius; ky <= radius; ++ky){
      for(int kx = -radius; kx <= radius; ++kx){
        const int source_x = clamp_index(2 * x + kx, input_width);
        const int source_y = clamp_index(2 * y + ky, input_height);
        const std::size_t input_index = (static_cast<std::size_t>(source_y) * input_width + source_x) * channels + channel;
        value += weights[ky + radius] * weights[kx + radius] * input[input_index];
      }
    }
    output[output_index + channel] = value;
  }
}

__global__ void expand_kernel(const float *input, float *output,
                              int input_width, int input_height, int channels,
                              int output_width, int output_height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= output_width || y >= output_height){
    return;
  }

  const float source_x = output_width == 1 ? 0.0f :
    static_cast<float>(x) * (input_width - 1) / (output_width - 1);
  const float source_y = output_height == 1 ? 0.0f :
    static_cast<float>(y) * (input_height - 1) / (output_height - 1);
  const int x0 = static_cast<int>(floorf(source_x));
  const int y0 = static_cast<int>(floorf(source_y));
  const int x1 = min(x0 + 1, input_width - 1);
  const int y1 = min(y0 + 1, input_height - 1);
  const float fx = source_x - x0;
  const float fy = source_y - y0;
  const std::size_t output_index = (static_cast<std::size_t>(y) * output_width + x) * channels;

  for(int channel = 0; channel < channels; ++channel){
    const float top_left = input[(static_cast<std::size_t>(y0) * input_width + x0) * channels + channel];
    const float top_right = input[(static_cast<std::size_t>(y0) * input_width + x1) * channels + channel];
    const float bottom_left = input[(static_cast<std::size_t>(y1) * input_width + x0) * channels + channel];
    const float bottom_right = input[(static_cast<std::size_t>(y1) * input_width + x1) * channels + channel];
    const float top = top_left + fx * (top_right - top_left);
    const float bottom = bottom_left + fx * (bottom_right - bottom_left);
    output[output_index + channel] = top + fy * (bottom - top);
  }
}

__global__ void subtract_kernel(const float *left, const float *right, float *output, std::size_t count)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if(index < count){
    output[index] = left[index] - right[index];
  }
}

__global__ void blend_kernel(const float *left, const float *right, const float *mask,
                             float *output, int width, int height, int channels)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= width || y >= height){
    return;
  }
  const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
  const float mask_value = mask[pixel];
  const std::size_t index = pixel * channels;
  for(int channel = 0; channel < channels; ++channel){
    output[index + channel] = mask_value * left[index + channel] +
                              (1.0f - mask_value) * right[index + channel];
  }
}

__global__ void add_kernel(const float *expanded, const float *laplacian,
                           float *output, std::size_t count)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if(index < count){
    output[index] = expanded[index] + laplacian[index];
  }
}

}

bool blend_images_cuda(const FloatImage &image_a,
                       const FloatImage &image_b,
                       const FloatImage &mask,
                       const std::vector<float> &weights,
                       int max_levels,
                       FloatImage &output,
                       int minimum_side,
                       CudaTiming *timing)
{
  if(image_a.width != image_b.width || image_a.height != image_b.height || image_a.channels != image_b.channels ||
     mask.width != image_a.width || mask.height != image_a.height || mask.channels != 1 ||
     weights.empty() || weights.size() % 2 == 0 || max_levels <= 0 || minimum_side <= 0){
    fprintf(stderr, "invalid CUDA blend inputs\n");
    return false;
  }

  std::vector<std::pair<int, int>> dimensions;
  dimensions.emplace_back(image_a.width, image_a.height);
  while(static_cast<int>(dimensions.size()) < max_levels){
    const auto [width, height] = dimensions.back();
    const int next_width = (width + 1) / 2;
    const int next_height = (height + 1) / 2;
    if(std::min(next_width, next_height) < minimum_side){
      break;
    }
    dimensions.emplace_back(next_width, next_height);
  }

  std::vector<DeviceLevel> levels(dimensions.size());
  float *device_weights = nullptr;
  float *collapse_a = nullptr;
  float *collapse_b = nullptr;
  cudaEvent_t transfer_start = nullptr, transfer_stop = nullptr;
  cudaEvent_t kernel_start = nullptr, kernel_stop = nullptr;

  auto cleanup = [&]() {
    for(DeviceLevel &level : levels){
      CUDA_CLEANUP(cudaFree(level.gaussian_a));
      CUDA_CLEANUP(cudaFree(level.gaussian_b));
      CUDA_CLEANUP(cudaFree(level.gaussian_mask));
      CUDA_CLEANUP(cudaFree(level.laplacian_a));
      CUDA_CLEANUP(cudaFree(level.laplacian_b));
      CUDA_CLEANUP(cudaFree(level.blended));
      CUDA_CLEANUP(cudaFree(level.expanded));
    }
    CUDA_CLEANUP(cudaFree(device_weights));
    CUDA_CLEANUP(cudaFree(collapse_a));
    CUDA_CLEANUP(cudaFree(collapse_b));
    CUDA_CLEANUP(cudaEventDestroy(transfer_start));
    CUDA_CLEANUP(cudaEventDestroy(transfer_stop));
    CUDA_CLEANUP(cudaEventDestroy(kernel_start));
    CUDA_CLEANUP(cudaEventDestroy(kernel_stop));
  };

  for(std::size_t index = 0; index < levels.size(); ++index){
    levels[index].width = dimensions[index].first;
    levels[index].height = dimensions[index].second;
    const std::size_t image_count = static_cast<std::size_t>(levels[index].width) * levels[index].height * image_a.channels;
    const std::size_t mask_count = static_cast<std::size_t>(levels[index].width) * levels[index].height;
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].gaussian_a), image_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].gaussian_b), image_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].gaussian_mask), mask_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].laplacian_a), image_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].laplacian_b), image_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].blended), image_count * sizeof(float)));
    CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&levels[index].expanded), image_count * sizeof(float)));
  }
  CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&device_weights), weights.size() * sizeof(float)));
  const std::size_t full_count = image_a.size();
  CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&collapse_a), full_count * sizeof(float)));
  CUDA_FAIL(cudaMalloc(reinterpret_cast<void **>(&collapse_b), full_count * sizeof(float)));
  CUDA_FAIL(cudaEventCreate(&transfer_start));
  CUDA_FAIL(cudaEventCreate(&transfer_stop));
  CUDA_FAIL(cudaEventCreate(&kernel_start));
  CUDA_FAIL(cudaEventCreate(&kernel_stop));

  CUDA_FAIL(cudaEventRecord(transfer_start));
  CUDA_FAIL(cudaMemcpy(levels[0].gaussian_a, image_a.pixels.data(), image_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_FAIL(cudaMemcpy(levels[0].gaussian_b, image_b.pixels.data(), image_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_FAIL(cudaMemcpy(levels[0].gaussian_mask, mask.pixels.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_FAIL(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_FAIL(cudaEventRecord(transfer_stop));
  CUDA_FAIL(cudaEventSynchronize(transfer_stop));
  if(timing != nullptr){
    CUDA_FAIL(cudaEventElapsedTime(&timing->host_to_device_ms, transfer_start, transfer_stop));
  }

  const dim3 block_2d(16, 16);
  const int radius = static_cast<int>(weights.size() / 2);
  CUDA_FAIL(cudaEventRecord(kernel_start));
  for(std::size_t level = 1; level < levels.size(); ++level){
    const DeviceLevel &previous = levels[level - 1];
    DeviceLevel &current = levels[level];
    const dim3 grid((current.width + block_2d.x - 1) / block_2d.x,
                    (current.height + block_2d.y - 1) / block_2d.y);
    gaussian_downsample_kernel<<<grid, block_2d>>>(previous.gaussian_a, current.gaussian_a,
                                                   previous.width, previous.height, image_a.channels,
                                                   current.width, current.height, device_weights, radius);
    CUDA_FAIL(cudaGetLastError());
    gaussian_downsample_kernel<<<grid, block_2d>>>(previous.gaussian_b, current.gaussian_b,
                                                   previous.width, previous.height, image_a.channels,
                                                   current.width, current.height, device_weights, radius);
    CUDA_FAIL(cudaGetLastError());
    gaussian_downsample_kernel<<<grid, block_2d>>>(previous.gaussian_mask, current.gaussian_mask,
                                                   previous.width, previous.height, 1,
                                                   current.width, current.height, device_weights, radius);
    CUDA_FAIL(cudaGetLastError());
  }

  for(std::size_t level = 0; level + 1 < levels.size(); ++level){
    DeviceLevel &current = levels[level];
    const DeviceLevel &next = levels[level + 1];
    const dim3 grid((current.width + block_2d.x - 1) / block_2d.x,
                    (current.height + block_2d.y - 1) / block_2d.y);
    expand_kernel<<<grid, block_2d>>>(next.gaussian_a, current.expanded, next.width, next.height,
                                      image_a.channels, current.width, current.height);
    CUDA_FAIL(cudaGetLastError());
    const std::size_t count = static_cast<std::size_t>(current.width) * current.height * image_a.channels;
    const int grid_1d = static_cast<int>((count + 255) / 256);
    subtract_kernel<<<grid_1d, 256>>>(current.gaussian_a, current.expanded, current.laplacian_a, count);
    CUDA_FAIL(cudaGetLastError());
    expand_kernel<<<grid, block_2d>>>(next.gaussian_b, current.expanded, next.width, next.height,
                                      image_a.channels, current.width, current.height);
    CUDA_FAIL(cudaGetLastError());
    subtract_kernel<<<grid_1d, 256>>>(current.gaussian_b, current.expanded, current.laplacian_b, count);
    CUDA_FAIL(cudaGetLastError());
  }
  CUDA_FAIL(cudaMemcpy(levels.back().laplacian_a, levels.back().gaussian_a,
                       static_cast<std::size_t>(levels.back().width) * levels.back().height * image_a.channels * sizeof(float),
                       cudaMemcpyDeviceToDevice));
  CUDA_FAIL(cudaMemcpy(levels.back().laplacian_b, levels.back().gaussian_b,
                       static_cast<std::size_t>(levels.back().width) * levels.back().height * image_a.channels * sizeof(float),
                       cudaMemcpyDeviceToDevice));

  for(std::size_t level = 0; level < levels.size(); ++level){
    DeviceLevel &current = levels[level];
    const dim3 grid((current.width + block_2d.x - 1) / block_2d.x,
                    (current.height + block_2d.y - 1) / block_2d.y);
    blend_kernel<<<grid, block_2d>>>(current.laplacian_a, current.laplacian_b, current.gaussian_mask,
                                      current.blended, current.width, current.height, image_a.channels);
    CUDA_FAIL(cudaGetLastError());
  }

  const DeviceLevel &coarse = levels.back();
  const std::size_t coarse_count = static_cast<std::size_t>(coarse.width) * coarse.height * image_a.channels;
  CUDA_FAIL(cudaMemcpy(collapse_a, coarse.blended, coarse_count * sizeof(float), cudaMemcpyDeviceToDevice));
  int current_width = coarse.width;
  int current_height = coarse.height;
  float *current_buffer = collapse_a;
  float *next_buffer = collapse_b;
  for(std::size_t index = levels.size() - 1; index-- > 0;){
    DeviceLevel &finer = levels[index];
    const dim3 grid((finer.width + block_2d.x - 1) / block_2d.x,
                    (finer.height + block_2d.y - 1) / block_2d.y);
    expand_kernel<<<grid, block_2d>>>(current_buffer, finer.expanded, current_width, current_height,
                                      image_a.channels, finer.width, finer.height);
    CUDA_FAIL(cudaGetLastError());
    const std::size_t count = static_cast<std::size_t>(finer.width) * finer.height * image_a.channels;
    const int grid_1d = static_cast<int>((count + 255) / 256);
    add_kernel<<<grid_1d, 256>>>(finer.expanded, finer.blended, next_buffer, count);
    CUDA_FAIL(cudaGetLastError());
    current_width = finer.width;
    current_height = finer.height;
    std::swap(current_buffer, next_buffer);
  }
  CUDA_FAIL(cudaEventRecord(kernel_stop));
  CUDA_FAIL(cudaEventSynchronize(kernel_stop));
  if(timing != nullptr){
    CUDA_FAIL(cudaEventElapsedTime(&timing->kernel_ms, kernel_start, kernel_stop));
  }

  output = FloatImage{image_a.width, image_a.height, image_a.channels, std::vector<float>(image_a.size())};
  CUDA_FAIL(cudaEventRecord(transfer_start));
  CUDA_FAIL(cudaMemcpy(output.pixels.data(), current_buffer, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_FAIL(cudaEventRecord(transfer_stop));
  CUDA_FAIL(cudaEventSynchronize(transfer_stop));
  if(timing != nullptr){
    CUDA_FAIL(cudaEventElapsedTime(&timing->device_to_host_ms, transfer_start, transfer_stop));
  }

  cleanup();
  return true;
}
