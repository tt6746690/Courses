CC=/usr/local/cuda-8.0/bin/nvcc

SOURCES=lab6.cpp

HEADERS=

INCLUDES+=-I/usr/local/cuda-8.0/include/ 
INCLUDES+=-I/usr/local-cuda-8.0/samples/common/inc/

CAPABILITY=-arch=sm_50

all: $(SOURCES)
	$(CC) $(SOURCES) $(INCLUDES) $(CAPABILITY) -O2 -o lab6.out
