CC = g++ -std=c++20
FLAGS = -fopenmp

SRCS = main.cpp \
			src/gaussian.cpp \
			src/gaussian_omp.cpp \
			src/pyramid.cpp \
			common/gaussian_common.cpp \
			common/image.cpp

CUDA = nvcc
CUDA_FLAGS = -std=c++20 -Xcompiler -fopenmp -DSEAMLESS_CUDA
CUDA_SRCS = main.cpp \
			src/gaussian.cpp \
			src/gaussian_omp.cpp \
			src/gaussian_cuda.cu \
			common/gaussian_common.cpp \
			common/image.cpp

seamless: $(SRCS)
		$(CC) $(FLAGS) -o seamless $(SRCS)

test_pyramid: src/pyramid.cpp tests/test_pyramid.cpp include/pyramid.hpp
		$(CC) $(FLAGS) -Iinclude -o test_pyramid src/pyramid.cpp tests/test_pyramid.cpp

seamless_cuda: $(CUDA_SRCS)
		$(CUDA) $(CUDA_FLAGS) -o seamless_cuda $(CUDA_SRCS)

clean:
		rm -f seamless seamless_cuda test_pyramid

.PHONY: clean seamless_cuda test_pyramid
