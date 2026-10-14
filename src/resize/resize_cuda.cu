#include "../../include/resize_cuda.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { \
  const cudaError_t error__ = (call); \
  if(error__ != cudaSuccess){ \
    fprintf(stderr, "CUDA error at %s:%d during %s: %s\\n", __FILE__, __LINE__, #call, cudaGetErrorString(error__)); \
    return false; \
  } \
} while(false)

namespace {

__global__ void resize_kernel(const float *input, float *output,
                              int width, int height, int channels,
                              int target_width, int target_height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= target_width || y >= target_height){
    return;
  }

  const float source_x = target_width == 1 ? 0.0f :
    static_cast<float>(x) * (width - 1) / (target_width - 1);
  const float source_y = target_height == 1 ? 0.0f :
    static_cast<float>(y) * (height - 1) / (target_height - 1);
  const int x0 = static_cast<int>(floorf(source_x));
  const int y0 = static_cast<int>(floorf(source_y));
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const float fx = source_x - x0;
  const float fy = source_y - y0;
  const std::size_t output_index = (static_cast<std::size_t>(y) * target_width + x) * channels;

  for(int channel = 0; channel < channels; ++channel){
    const float top_left = input[(static_cast<std::size_t>(y0) * width + x0) * channels + channel];
    const float top_right = input[(static_cast<std::size_t>(y0) * width + x1) * channels + channel];
    const float bottom_left = input[(static_cast<std::size_t>(y1) * width + x0) * channels + channel];
    const float bottom_right = input[(static_cast<std::size_t>(y1) * width + x1) * channels + channel];
    const float top = top_left + fx * (top_right - top_left);
    const float bottom = bottom_left + fx * (bottom_right - bottom_left);
    output[output_index + channel] = top + fy * (bottom - top);
  }
}

}

bool resize_bilinear_cuda(const FloatImage &input,
                          int target_width,
                          int target_height,
                          FloatImage &output,
                          CudaTiming *timing)
{
  if(input.width <= 0 || input.height <= 0 || input.channels <= 0 || target_width <= 0 || target_height <= 0){
    fprintf(stderr, "invalid CUDA resize dimensions\n");
    return false;
  }

  output = FloatImage{target_width, target_height, input.channels,
                      std::vector<float>(static_cast<std::size_t>(target_width) * target_height * input.channels)};
  float *device_input = nullptr;
  float *device_output = nullptr;
  cudaEvent_t transfer_start = nullptr, transfer_stop = nullptr;
  cudaEvent_t kernel_start = nullptr, kernel_stop = nullptr;
  bool success = false;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)));
  CUDA_CHECK(cudaEventCreate(&transfer_start));
  CUDA_CHECK(cudaEventCreate(&transfer_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  CUDA_CHECK(cudaEventRecord(transfer_start));
  CUDA_CHECK(cudaMemcpy(device_input, input.pixels.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(transfer_stop));
  CUDA_CHECK(cudaEventSynchronize(transfer_stop));
  if(timing != nullptr){
    CUDA_CHECK(cudaEventElapsedTime(&timing->host_to_device_ms, transfer_start, transfer_stop));
  }

  const dim3 block(16, 16);
  const dim3 grid((target_width + block.x - 1) / block.x, (target_height + block.y - 1) / block.y);
  CUDA_CHECK(cudaEventRecord(kernel_start));
  resize_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height, input.channels,
                                 target_width, target_height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));
  if(timing != nullptr){
    CUDA_CHECK(cudaEventElapsedTime(&timing->kernel_ms, kernel_start, kernel_stop));
  }

  CUDA_CHECK(cudaEventRecord(transfer_start));
  CUDA_CHECK(cudaMemcpy(output.pixels.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(transfer_stop));
  CUDA_CHECK(cudaEventSynchronize(transfer_stop));
  if(timing != nullptr){
    CUDA_CHECK(cudaEventElapsedTime(&timing->device_to_host_ms, transfer_start, transfer_stop));
  }
  success = true;

  CUDA_CHECK(cudaEventDestroy(transfer_start));
  CUDA_CHECK(cudaEventDestroy(transfer_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return success;
}
