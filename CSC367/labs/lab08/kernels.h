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

#ifndef __KERNELS__H
#define __KERNELS__H


#define NUM_KERNELS 10

constexpr int maxThreads = 512; // threads per block
constexpr int maxBlocks = 64;

// Attempt #1: interleaved addressing + divergent branch
__global__ void dot_kernel1(float *g_idata_1, float *g_idata2, float *g_odata);

// Attempt #2: interleaved addressing + bank conflicts
__global__ void dot_kernel2(float *g_idata_1, float *g_idata2, float *g_odata); 

// Attempt #3: sequential addressing
__global__ void dot_kernel3(float *g_idata_1, float *g_idata2, float *g_odata);

// Attempt #4: first add during global load
__global__ void dot_kernel4(float *g_idata_1, float *g_idata2, float *g_odata);

// Attempt #5: unroll the last warp
__global__ void dot_kernel5(float *g_idata_1, float *g_idata2, float *g_odata);

// Attempt #6: completely unrolled
template <unsigned int blockSize>
__global__ void dot_kernel6(float *g_idata_1, float *g_idata2, float *g_odata);
#include "dot_kernel6.h"

// Attempt #7: multiple adds per thread
template <unsigned int blockSize>
__global__ void dot_kernel7(float *g_idata_1, float *g_idata2, float *g_odata,
        unsigned int n);
#include "dot_kernel7.h"

// Attempt #8: shuffle instruction
__global__ void dot_kernel8(float *g_idata1, float *g_idata2, float *g_odata,
        int N);
// Attempt #9: shfl instructions + warp atomic
__global__ void dot_kernel9(float *g_idata1, float *g_idata2, float *g_odata,
        int N);
// Attempt #10: shfl instructions + block atomic
__global__ void dot_kernel10(float *g_idata1, float *g_idata2, float *g_odata,
        int N);
#endif
