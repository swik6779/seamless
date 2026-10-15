#include "../include/pyramid.hpp"
#include "../include/resize.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  FloatImage input{5, 3, 1, {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    10.0f, 11.0f, 12.0f, 13.0f, 14.0f
  }};

  const std::vector<float> identity_weights{1.0f};
  const FloatImage reduced = downsample(input, identity_weights);
  assert(reduced.width == 3);
  assert(reduced.height == 2);
  assert(reduced.pixels[0] == 0.0f);
  assert(reduced.pixels[1] == 2.0f);
  assert(reduced.pixels[2] == 4.0f);
  assert(reduced.pixels[3] == 10.0f);
  assert(reduced.pixels[4] == 12.0f);
  assert(reduced.pixels[5] == 14.0f);

  const FloatImage expanded = expand_bilinear(reduced, 5, 3);
  assert(expanded.width == 5);
  assert(expanded.height == 3);
  assert(expanded.channels == 1);
  assert(std::fabs(expanded.pixels.front() - 0.0f) < 1e-6f);
  assert(std::fabs(expanded.pixels.back() - 14.0f) < 1e-6f);

  const FloatImage resized = resize_bilinear(input, 7, 5);
  const FloatImage resized_omp = resize_bilinear_omp(input, 7, 5);
  assert(resized.width == 7 && resized.height == 5 && resized.channels == 1);
  assert(max_absolute_difference(resized, resized_omp) < 1e-6f);

  const FloatImage one_row = resize_bilinear(FloatImage{1, 3, 1, {1.0f, 2.0f, 3.0f}}, 4, 1);
  assert(one_row.width == 4 && one_row.height == 1);
  for(float value : one_row.pixels){
    assert(std::fabs(value - 1.0f) < 1e-6f);
  }

  const FloatImage single_pixel = expand_bilinear(FloatImage{1, 1, 1, {3.5f}}, 4, 3);
  for(float value : single_pixel.pixels){
    assert(std::fabs(value - 3.5f) < 1e-6f);
  }

  FloatImage gradient{33, 33, 3,
                      std::vector<float>(static_cast<std::size_t>(33) * 33 * 3)};
  for(int y = 0; y < gradient.height; ++y){
    for(int x = 0; x < gradient.width; ++x){
      const std::size_t index = (static_cast<std::size_t>(y) * gradient.width + x) * gradient.channels;
      gradient.pixels[index] = static_cast<float>(x) / gradient.width;
      gradient.pixels[index + 1] = static_cast<float>(y) / gradient.height;
      gradient.pixels[index + 2] = static_cast<float>(x + y) / (gradient.width + gradient.height);
    }
  }

  const std::vector<float> blur_weights{0.25f, 0.5f, 0.25f};
  const ImagePyramid gaussian = build_gaussian_pyramid(gradient, blur_weights, 6);
  assert(gaussian.size() == 2);
  assert(gaussian[0].width == 33 && gaussian[0].height == 33);
  assert(gaussian[1].width == 17 && gaussian[1].height == 17);

  const ImagePyramid laplacian = build_laplacian_pyramid(gaussian);
  const FloatImage reconstructed = collapse_laplacian_pyramid(laplacian);
  const float round_trip_error = max_absolute_difference(gradient, reconstructed);
  assert(round_trip_error < 1e-6f);
  std::cout << "round-trip max abs error: " << round_trip_error << "\n";

  FloatImage other = gradient;
  for(float &value : other.pixels){
    value = 1.0f - value;
  }
  FloatImage zero_mask{33, 33, 1, std::vector<float>(static_cast<std::size_t>(33) * 33, 0.0f)};
  FloatImage one_mask{33, 33, 1, std::vector<float>(static_cast<std::size_t>(33) * 33, 1.0f)};
  FloatImage half_mask{33, 33, 1, std::vector<float>(static_cast<std::size_t>(33) * 33, 0.5f)};

  assert(max_absolute_difference(blend_images_cpu(gradient, other, zero_mask, blur_weights, 6), other) < 1e-6f);
  assert(max_absolute_difference(blend_images_cpu(gradient, other, one_mask, blur_weights, 6), gradient) < 1e-6f);
  const FloatImage cpu_half = blend_images_cpu(gradient, other, half_mask, blur_weights, 6);
  const FloatImage omp_half = blend_images_omp(gradient, other, half_mask, blur_weights, 6);
  assert(max_absolute_difference(cpu_half, omp_half) < 1e-6f);

  StageTiming cpu_timing;
  StageTiming omp_timing;
  const FloatImage timed_cpu = blend_images_cpu(gradient, other, half_mask, blur_weights, 6, 16, &cpu_timing);
  const FloatImage timed_omp = blend_images_omp(gradient, other, half_mask, blur_weights, 6, 16, &omp_timing);
  assert(max_absolute_difference(timed_cpu, cpu_half) < 1e-6f);
  assert(max_absolute_difference(timed_omp, omp_half) < 1e-6f);
  assert(cpu_timing.blur_ms >= 0.0 && cpu_timing.downsample_ms >= 0.0 &&
         cpu_timing.expand_ms >= 0.0 && cpu_timing.subtract_ms >= 0.0 &&
         cpu_timing.blend_ms >= 0.0 && cpu_timing.collapse_ms >= 0.0);
  assert(omp_timing.blur_ms >= 0.0 && omp_timing.downsample_ms >= 0.0 &&
         omp_timing.expand_ms >= 0.0 && omp_timing.subtract_ms >= 0.0 &&
         omp_timing.blend_ms >= 0.0 && omp_timing.collapse_ms >= 0.0);
  std::cout << "CPU/OpenMP blend max abs difference: "
            << max_absolute_difference(cpu_half, omp_half) << "\n";

  std::cout << "pyramid primitive tests passed\n";
  return 0;
}
