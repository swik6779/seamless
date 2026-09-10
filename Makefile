CC = g++ -std=c++20

FLAGS = -fopenmp

SRCS = main.cpp \
				src/gaussian.cpp

seamless: $(SRCS)
		$(CC) $(FLAGS) -o seamless $(SRCS)

clean:
		rm -f seamless

.PHONY: clean
