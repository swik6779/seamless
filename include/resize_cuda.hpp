#ifndef RESIZE_CUDA_HPP
#define RESIZE_CUDA_HPP

#include "pyramid.hpp"

struct CudaTiming {
  float host_to_device_ms = 0.0f;
  float device_to_host_ms = 0.0f;
  float kernel_ms = 0.0f;
};

bool resize_bilinear_cuda(const FloatImage &input,
                          int target_width,
                          int target_height,
                          FloatImage &output,
                          CudaTiming *timing = nullptr);

#endif
