CUDAHOME=/pkgs_local/cuda-10.0
# CC=/usr/local/cuda-8.0/bin/nvcc
CC=$(CUDAHOME)/bin/nvcc


SOURCES=lab6.cpp

HEADERS=

INCLUDES+=-I$(CUDAHOME)/include/ 
INCLUDES+=-I$(CUDAHOME)/samples/common/inc/

CAPABILITY=-arch=sm_50

all: $(SOURCES)
	$(CC) $(SOURCES) $(INCLUDES) $(CAPABILITY) -O2 -o lab6.out
