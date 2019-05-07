CC=/pkgs_local/cuda-10.0/bin/nvcc
CAPABILITY=-arch=sm_50
SRCS = $(wildcard *.cu)
PROGS = $(patsubst %.cu,%,$(SRCS))

all: $(PROGS)

%: %.cu
	$(CC) $(CAPABILITY) -o $@ $<

clean:
	rm -f $(PROGS)
