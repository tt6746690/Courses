CC=/usr/local/cuda-8.0/bin/nvcc

CAPABILITY=-arch=sm_50
GCC_OPT = -O2 -std=c++11 $(CAPABILITY) -Xcompiler -Wall
OBJ = lab8.o dot_kernel1.o  dot_kernel2.o dot_kernel3.o dot_kernel4.o \
	dot_kernel5.o dot_kernel8-10.o
DEPS = kernels.h dot_kernel6.h dot_kernel7.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(GCC_OPT)

all: $(OBJ)
	$(CC) $(GCC_OPT) $^ -o lab8.out
	
clean:
	rm -f *.o *.out
