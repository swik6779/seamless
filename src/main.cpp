#include "../include/gaussian.hpp"

void print_matrix(std::vector<std::vector<double>> mat){
  for(std::vector<double> arr:mat){
    for(double d:arr){
      std::cout << d << "\t";
    }
    std::cout << "\n";
  }
}

int main(){
  int k = 5;
  std::vector<std::vector<double>> lm = create_gaussian_kernel(k);

  double total = 0.0;
  for(std::vector<double> arr:lm){
    for(double d:arr){
      total += d;
    }
  }
  print_matrix(lm);
  std::cout << "total: " << total;

  return 0;
}
