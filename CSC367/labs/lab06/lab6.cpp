/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define REPLACE_ME 0

int main( int argc, char** argv) {

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n", 
		       (int)error_id, 
		       cudaGetErrorString(error_id));
		exit(1);
	}
	// This returns 0 if there's no CUDA-capable device
	if (deviceCount == 0)
		printf("There is no device supporting CUDA\n");
	else
		printf("Found %d CUDA Capable device(s)\n", deviceCount);

	int dev;
	for(dev = 0; dev < deviceCount; dev++) {

		struct cudaDeviceProp p;
		cudaGetDeviceProperties(&p, dev);

		printf("\nDevice %d: \"%s\"\n", dev, p.name);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", 
		       p.major, p.minor);
		printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP: %d CUDA Cores\n",
		       p.multiProcessorCount,
		       _ConvertSMVer2Cores(p.major, p.minor),
               	p.multiProcessorCount *  _ConvertSMVer2Cores(p.major, p.minor));
		printf("  GPU Clock Rate:                                %.2f GHz\n", 
		       p.clockRate * 1e-6f);

		printf("  Memory Clock Rate:                             %.2f GHz\n", 
		       p.memoryClockRate * 1e-6f);

		printf("  Memory Bus Width:                              %d-bit\n", 
		       p.memoryBusWidth);

        printf("\n"); printf("  Total amount of global memory:"
                "%.0f MBytes (%llu bytes)\n",
                (float) p.totalGlobalMem /1048576.0f, 
		       (unsigned long long) p.totalGlobalMem);

		printf("  Shared Memory per Block:                       %ld\n", 
		       p.sharedMemPerBlock);

		printf("  L2 Cache Size:                                 %d\n", 
		       p.l2CacheSize);

		printf("  Registers per Block:                           %d\n", 
		       p.regsPerBlock);

		printf("\n");
        printf("  Max grid size:                                 (%d, %d,"
            "%d)\n", 
		       p.maxGridSize[0], 
		       p.maxGridSize[1], 
		       p.maxGridSize[2]);

        printf("  Max thread dimensions:                         (%d, %d,"
            "%d)\n", 
		       p.maxThreadsDim[0], 
		       p.maxThreadsDim[1], 
		       p.maxThreadsDim[2]);

		printf("  Max threads per block:                         %d\n", 
		       p.maxThreadsPerBlock);

		printf("  Warp size:                                     %d\n", 
		       p.warpSize);
	
		printf("\n");
	}

	return 0;
}
