#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include <cstddef>
#include <vector>

struct FloatImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> pixels;

  std::size_t size() const;
};

struct StageTiming {
  double blur_ms = 0.0;
  double downsample_ms = 0.0;
  double expand_ms = 0.0;
  double subtract_ms = 0.0;
  double blend_ms = 0.0;
  double collapse_ms = 0.0;
};

using ImagePyramid = std::vector<FloatImage>;

// All coordinates use clamp-to-edge borders and row-major interleaved channels.
FloatImage downsample(const FloatImage &input, const std::vector<float> &weights);
FloatImage blur_float_cpu(const FloatImage &input, const std::vector<float> &weights);
FloatImage downsample_from_blurred(const FloatImage &blurred);
FloatImage expand_bilinear(const FloatImage &input, int target_width, int target_height);
ImagePyramid build_gaussian_pyramid(const FloatImage &input,
                                    const std::vector<float> &weights,
                                    int max_levels,
                                    int minimum_side = 16);
ImagePyramid build_laplacian_pyramid(const ImagePyramid &gaussian);
FloatImage collapse_laplacian_pyramid(const ImagePyramid &laplacian);
float max_absolute_difference(const FloatImage &left, const FloatImage &right);
FloatImage blend_images_cpu(const FloatImage &image_a,
                            const FloatImage &image_b,
                            const FloatImage &mask,
                            const std::vector<float> &weights,
                            int max_levels,
                            int minimum_side = 16,
                            StageTiming *timing = nullptr);
FloatImage blend_images_omp(const FloatImage &image_a,
                            const FloatImage &image_b,
                            const FloatImage &mask,
                            const std::vector<float> &weights,
                            int max_levels,
                            int minimum_side = 16,
                            StageTiming *timing = nullptr);

#endif
