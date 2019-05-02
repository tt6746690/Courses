CC=/pkgs_local/cuda-10.0/bin/nvcc

SOURCES=lab7.cu

CAPABILITY=-arch=sm_50

all: $(SOURCES)
	$(CC) $(SOURCES) $(CAPABILITY) -O2 -o lab7.out
