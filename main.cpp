#include "include/gaussian.hpp"
#include <string.h>
#include <omp.h>

/*
  We expect the user to type something like
  seamless --kernel_size=k --file=filename.png
*/
int main(int argc, char **argv){

  if(argc < 3){
    fprintf(stderr, "Incorrect usage\n");
    return 0;
  }
  
  char *kernel_specification = argv[1];
  char *filename_specification = argv[2];

  std::vector<std::vector<float>> kernel = create_gaussian_kernel(kernel_specification[strlen(kernel_specification) - 1] - '0');
  double naive_start = omp_get_wtime();
  gaussian_blur_iterable_naive(filename_specification, kernel);
  double naive_end = omp_get_wtime();
  
  std::vector<float> weights = create_gaussian_weights(kernel_specification[strlen(kernel_specification) - 1] - '0');
  double seperable_start = omp_get_wtime();
  gaussian_blur_iterable_naive(filename_specification, kernel);
  double seperable_end = omp_get_wtime();

  printf("\nNaive Time: %lf", (naive_end - naive_start));
  printf("\nIterable Seperable Time: %lf", (seperable_end - seperable_start));
  return 0;
}
