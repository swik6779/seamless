CC = g++ -std=c++20

SRCS = main.cpp \
				src/gaussian.cpp

seamless: $(SRCS)
		$(CC) -o seamless $(SRCS)

clean:
		rm -f seamless

.PHONY: clean
