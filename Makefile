CC = g++
FLAGS = -fopenmp

SRCS = main.cpp \
			src/gaussian_blur/gaussian.cpp \
			src/gaussian_blur/gaussian_omp.cpp \
			src/pyramid/pyramid.cpp \
			src/laplacian_blend/blend.cpp \
			src/laplacian_blend/blend_omp.cpp \
			src/resize/resize.cpp \
			src/resize/resize_omp.cpp \
			src/pyramid_io.cpp \
			common/gaussian_common.cpp \
			common/image.cpp

CUDA = nvcc
CUDA_FLAGS = -Xcompiler -fopenmp -DSEAMLESS_CUDA
CUDA_SRCS = main.cpp \
			src/gaussian_blur/gaussian.cpp \
			src/gaussian_blur/gaussian_omp.cpp \
			src/pyramid/pyramid.cpp \
			src/laplacian_blend/blend.cpp \
			src/laplacian_blend/blend_omp.cpp \
			src/resize/resize.cpp \
			src/resize/resize_omp.cpp \
			src/resize/resize_cuda.cu \
			src/pyramid_io.cpp \
			src/gaussian_blur/gaussian_cuda.cu \
			src/laplacian_blend/blend_cuda.cu \
			common/gaussian_common.cpp \
			common/image.cpp

seamless: $(SRCS)
		$(CC) $(FLAGS) -o seamless $(SRCS)

test_pyramid: src/pyramid/pyramid.cpp src/laplacian_blend/blend.cpp src/laplacian_blend/blend_omp.cpp src/resize/resize.cpp src/resize/resize_omp.cpp src/pyramid_io.cpp tests/test_pyramid.cpp include/pyramid.hpp include/pyramid_io.hpp include/resize.hpp
		$(CC) $(FLAGS) -Iinclude -o test_pyramid src/pyramid/pyramid.cpp src/laplacian_blend/blend.cpp src/laplacian_blend/blend_omp.cpp src/resize/resize.cpp src/resize/resize_omp.cpp src/pyramid_io.cpp tests/test_pyramid.cpp common/image.cpp

seamless_cuda: $(CUDA_SRCS)
		$(CUDA) $(CUDA_FLAGS) -o seamless_cuda $(CUDA_SRCS)

clean:
		rm -f seamless seamless_cuda test_pyramid

.PHONY: clean seamless_cuda test_pyramid
