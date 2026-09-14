CC = g++-15 -std=c++20
FLAGS = -fopenmp

SRCS = main.cpp \
			src/gaussian.cpp

OMP_SRCS = main_omp.cpp \
			src/gaussian_omp.cpp \
			src/gaussian.cpp

seamless: $(SRCS)
		$(CC) $(FLAGS) -o seamless $(SRCS)

seamless_omp: $(OMP_SRCS)
		$(CC) $(FLAGS) -o seamless_omp $(OMP_SRCS)

clean:
		rm -f seamless seamless_omp

.PHONY: clean
