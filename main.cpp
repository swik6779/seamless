#include "include/gaussian.hpp"
#include <string.h>

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
  
  std::vector<std::vector<double>> kernel = create_gaussian_kernel(kernel_specification[strlen(kernel_specification) - 1] - '0');
  gaussian_blur(filename_specification, kernel);
  
  return 0;
}
