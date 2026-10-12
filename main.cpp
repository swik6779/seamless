#include "include/gaussian.hpp"
#include "include/gaussian_omp.hpp"
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
