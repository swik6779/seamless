#include "include/gaussian.hpp"
#include "include/gaussian_omp.hpp"
#include <string.h>
#include <omp.h>

/*
  Usage:
  seamless_omp --kernel_size=k filename.png
*/
int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "Incorrect usage\nExpected: seamless_omp --kernel_size=<k> <file>\n");
    return 1;
  }

  char *kernel_specification  = argv[1];
  char *filename_specification = argv[2];

  int k = kernel_specification[strlen(kernel_specification) - 1] - '0';

  std::vector<std::vector<float>> kernel  = create_gaussian_kernel(k);
  std::vector<float>              weights = create_gaussian_weights(k);

  printf("Running with %d OpenMP thread(s)\n\n", omp_get_max_threads());

  // --- OMP Naive ---
  double omp_naive_start = omp_get_wtime();
  gaussian_blur_omp_naive(filename_specification, kernel);
  double omp_naive_end   = omp_get_wtime();

  // --- OMP Separable ---
  double omp_sep_start = omp_get_wtime();
  gaussian_blur_omp_separable(filename_specification, weights);
  double omp_sep_end   = omp_get_wtime();

  printf("OMP Naive Time:      %lf seconds\n", (omp_naive_end - omp_naive_start));
  printf("OMP Separable Time:  %lf seconds\n", (omp_sep_end   - omp_sep_start));

  return 0;
}
