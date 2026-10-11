#include "include/gaussian.hpp"
#include "include/gaussian_omp.hpp"
#ifdef SEAMLESS_CUDA
#include "include/gaussian_cuda.hpp"
#endif
#include <string.h>
#include <omp.h>

/*
  We expect the user to type something like
  seamless --kernel_size=k --file=filename.png --exec_mode=<itna/itsep/ompna/ompsep/cuna/cusep> --output_file_name
*/

int main(int argc, char **argv){

  if(argc < 3){
    fprintf(stderr, "Incorrect usage\n");
    return 0;
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
