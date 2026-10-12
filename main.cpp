#include "include/gaussian.hpp"
#include "include/gaussian_omp.hpp"
#include "include/pyramid.hpp"
#include "include/pyramid_io.hpp"
#include "include/resize.hpp"
#ifdef SEAMLESS_CUDA
#include "include/gaussian_cuda.hpp"
#endif
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <omp.h>

/*
  We expect the user to type something like
  seamless --kernel_size=k --file=filename.png --exec_mode=<itna/itsep/ompna/ompsep/cuna/cusep> --output_file_name
*/

namespace {

const char *option_value(int argc, char **argv, const char *name)
{
  const std::size_t name_length = strlen(name);
  for(int i = 2; i + 1 < argc; ++i){
    if(strcmp(argv[i], name) == 0){
      return argv[i + 1];
    }
  }
  for(int i = 2; i < argc; ++i){
    if(strncmp(argv[i], name, name_length) == 0 && argv[i][name_length] == '='){
      return argv[i] + name_length + 1;
    }
  }
  return nullptr;
}

int integer_option(int argc, char **argv, const char *name, int default_value)
{
  const char *value = option_value(argc, argv, name);
  return value == nullptr ? default_value : std::atoi(value);
}

int run_resize(int argc, char **argv)
{
  const char *input_path = option_value(argc, argv, "--input");
  const char *output_path = option_value(argc, argv, "--output");
  const char *backend = option_value(argc, argv, "--backend");
  const char *width_text = option_value(argc, argv, "--width");
  const char *height_text = option_value(argc, argv, "--height");
  const bool has_width = width_text != nullptr;
  const bool has_height = height_text != nullptr;
  const int requested_width = has_width ? std::atoi(width_text) : 0;
  const int requested_height = has_height ? std::atoi(height_text) : 0;

  if(input_path == nullptr || output_path == nullptr || backend == nullptr || has_width == has_height ||
     (has_width && requested_width <= 0) || (has_height && requested_height <= 0)){
    fprintf(stderr, "usage: seamless resize --input FILE --output FILE "
                    "(--width W | --height H) --backend cpu|omp\n");
    return 2;
  }

  Image loaded = load_image(input_path);
  if(loaded.pixels == nullptr){
    return 1;
  }

  try{
    const FloatImage input = image_to_float(loaded);
    const double aspect = static_cast<double>(input.width) / input.height;
    const int target_width = has_width ? requested_width :
      std::max(1, static_cast<int>(std::lround(requested_height * aspect)));
    const int target_height = has_height ? requested_height :
      std::max(1, static_cast<int>(std::lround(requested_width / aspect)));
    const double start = omp_get_wtime();
    FloatImage output;
    if(strcmp(backend, "cpu") == 0){
      output = resize_bilinear(input, target_width, target_height);
    }
    else if(strcmp(backend, "omp") == 0){
      output = resize_bilinear_omp(input, target_width, target_height);
    }
    else{
      fprintf(stderr, "unsupported resize backend: %s\n", backend);
      free_image(&loaded);
      return 2;
    }

    const bool written = write_float_image(output_path, output);
    fprintf(stdout, "%s (%dx%d)\nElapsed time: %.6f seconds\n",
            written ? "Resize output saved" : "Could not write resize output",
            target_width, target_height, omp_get_wtime() - start);
    free_image(&loaded);
    return written ? 0 : 1;
  }
  catch(const std::exception &error){
    fprintf(stderr, "resize failed: %s\n", error.what());
    free_image(&loaded);
    return 1;
  }
}

int run_laplacian_blend(int argc, char **argv)
{
  const char *path_a = option_value(argc, argv, "--a");
  const char *path_b = option_value(argc, argv, "--b");
  const char *path_mask = option_value(argc, argv, "--mask");
  const char *output = option_value(argc, argv, "--output");
  const char *backend = option_value(argc, argv, "--backend");
  const int kernel_size = integer_option(argc, argv, "--kernel-size", 7);
  const int levels = integer_option(argc, argv, "--levels", 6);

  if(path_a == nullptr || path_b == nullptr || path_mask == nullptr || output == nullptr || backend == nullptr ||
     kernel_size <= 0 || kernel_size % 2 == 0 || levels <= 0){
    fprintf(stderr, "usage: seamless laplac-blend --a FILE --b FILE --mask FILE --output FILE "
                    "--backend cpu|omp --kernel-size ODD --levels N\n");
    return 2;
  }

  Image loaded_a = load_image(path_a);
  Image loaded_b = load_image(path_b);
  Image loaded_mask = load_image(path_mask);
  if(loaded_a.pixels == nullptr || loaded_b.pixels == nullptr || loaded_mask.pixels == nullptr){
    free_image(&loaded_a);
    free_image(&loaded_b);
    free_image(&loaded_mask);
    return 1;
  }

  try{
    const FloatImage image_a = image_to_float(loaded_a);
    FloatImage image_b = image_to_float(loaded_b);
    FloatImage mask = mask_to_float(loaded_mask);
    if(image_b.width != image_a.width || image_b.height != image_a.height){
      image_b = strcmp(backend, "omp") == 0
        ? resize_bilinear_omp(image_b, image_a.width, image_a.height)
        : resize_bilinear(image_b, image_a.width, image_a.height);
    }
    if(mask.width != image_a.width || mask.height != image_a.height){
      mask = strcmp(backend, "omp") == 0
        ? resize_bilinear_omp(mask, image_a.width, image_a.height)
        : resize_bilinear(mask, image_a.width, image_a.height);
    }
    const auto weights = create_gaussian_weights(kernel_size);
    const double start = omp_get_wtime();
    FloatImage result;
    if(strcmp(backend, "cpu") == 0){
      result = blend_images_cpu(image_a, image_b, mask, weights, levels);
    }
    else if(strcmp(backend, "omp") == 0){
      result = blend_images_omp(image_a, image_b, mask, weights, levels);
    }
    else{
      fprintf(stderr, "unsupported blending backend: %s\n", backend);
      free_image(&loaded_a);
      free_image(&loaded_b);
      free_image(&loaded_mask);
      return 2;
    }

    const bool written = write_float_image(output, result);
    fprintf(stdout, "%s\nElapsed time: %.6f seconds\n",
            written ? "Blend output saved" : "Could not write blend output",
            omp_get_wtime() - start);
    free_image(&loaded_a);
    free_image(&loaded_b);
    free_image(&loaded_mask);
    return written ? 0 : 1;
  }
  catch(const std::exception &error){
    fprintf(stderr, "blend failed: %s\n", error.what());
    free_image(&loaded_a);
    free_image(&loaded_b);
    free_image(&loaded_mask);
    return 1;
  }
}

int run_gaussian_blur(int argc, char **argv)
{
  const char *input = option_value(argc, argv, "--input");
  const char *output = option_value(argc, argv, "--output");
  const char *backend = option_value(argc, argv, "--backend");
  const int kernel_size = integer_option(argc, argv, "--kernel-size", 7);

  if(input == nullptr || output == nullptr || backend == nullptr || kernel_size <= 0 || kernel_size % 2 == 0){
    fprintf(stderr, "usage: seamless gauss-blur --input FILE --output FILE --backend cpu|omp|cuda --kernel-size ODD\n");
    return 2;
  }

  const double start = omp_get_wtime();
  if(strcmp(backend, "cpu") == 0){
    auto kernel = create_gaussian_kernel(kernel_size);
    gaussian_blur_iterable_naive(input, kernel, output);
  }
  else if(strcmp(backend, "omp") == 0){
    auto weights = create_gaussian_weights(kernel_size);
    gaussian_blur_omp_separable(input, weights, output);
  }
#ifdef SEAMLESS_CUDA
  else if(strcmp(backend, "cuda") == 0){
    auto weights = create_gaussian_weights(kernel_size);
    gaussian_blur_cuda_separable(input, weights, output);
  }
#endif
  else{
    fprintf(stderr, "unsupported backend: %s\n", backend);
    return 2;
  }

  fprintf(stdout, "Output saved to %s\nElapsed time: %.6f seconds\n", output, omp_get_wtime() - start);
  return 0;
}

}

int main(int argc, char **argv){

  if(argc < 2){
    fprintf(stderr, "usage: seamless gauss-blur ...\n");
    return 2;
  }

  if(strcmp(argv[1], "gauss-blur") == 0){
    return run_gaussian_blur(argc, argv);
  }
  if(strcmp(argv[1], "laplac-blend") == 0){
    return run_laplacian_blend(argc, argv);
  }
  if(strcmp(argv[1], "resize") == 0){
    return run_resize(argc, argv);
  }

  // Preserve the original modes while the remaining subcommands are added.
  if(argc < 5){
    fprintf(stderr, "unknown command or incomplete legacy invocation\n");
    return 2;
  }

  char *kernel_specification = argv[1];
  char *filename_specification = argv[2];
  char *exec_mode_specification = argv[3];
  char *output_filename = argv[4];

  double start, end;
  
  if(strcmp(exec_mode_specification, "itna") == 0){
    std::vector<std::vector<float>> kernel = create_gaussian_kernel(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_iterable_naive(filename_specification, kernel, output_filename);
    end = omp_get_wtime();
  }
  else if(strcmp(exec_mode_specification, "itsep") == 0){
    std::vector<float> weights = create_gaussian_weights(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_iterable_seperable(filename_specification, weights, output_filename);
    end = omp_get_wtime();
  }
  
  else if(strcmp(exec_mode_specification, "ompna") == 0){
    std::vector<std::vector<float>> kernel = create_gaussian_kernel(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_omp_naive(filename_specification, kernel, output_filename);
    end = omp_get_wtime();
  }
  else if(strcmp(exec_mode_specification, "ompsep") == 0){
    std::vector<float> weights = create_gaussian_weights(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_omp_separable(filename_specification, weights, output_filename);
    end = omp_get_wtime();
  }
#ifdef SEAMLESS_CUDA
  else if(strcmp(exec_mode_specification, "cuna") == 0){
    std::vector<std::vector<float>> kernel = create_gaussian_kernel(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_cuda_naive(filename_specification, kernel, output_filename);
    end = omp_get_wtime();
  }
  else if(strcmp(exec_mode_specification, "cusep") == 0){
    std::vector<float> weights = create_gaussian_weights(kernel_specification[strlen(kernel_specification) - 1] - '0');
    start = omp_get_wtime();
    gaussian_blur_cuda_separable(filename_specification, weights, output_filename);
    end = omp_get_wtime();
  }
#endif

  else{
    fprintf(stderr, "incorrect mode specified\n");
    exit(1);
  }

  fprintf(stdout, "\nOutput saved to %s", output_filename);
  fprintf(stdout, "\nComplete execution time: %lf (method: %s)\n", (end - start), exec_mode_specification);
  
  return 0;
}
