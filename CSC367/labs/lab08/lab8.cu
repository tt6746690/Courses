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

#include <cuda.h>
#include <stdio.h>
#include <limits.h>

#include "kernels.h"
#include "cpu_dot.h"

//-----------------------------------------
//--- constants, macros, definitions ------
//-----------------------------------------
#define M (1024*1024)
#define M2      (2*M)
#define M8      (8*M)
#define M32    (32*M)

#define NUM_ITERATIONS 100 // Change this to run several iterations and get average
#define MAX_BLOCK_SIZE 65535
#define TO_SECONDS 1000
#define MY_MIN(x,y) ((x < y) ? x : y)

template <unsigned int blockSize>
void call_kernel_dot(int kernel, const dim3 &dimGrid, const dim3 &dimBlock,
        int shMemSize, float *d_idata1, float *d_idata2, float *d_odata,
        int size)
{
    switch (kernel)
    {
        case 1:
        {
            dot_kernel1<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata);
            break;
        }
        case 2:
        {
            dot_kernel2<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata);
            break;
        }
        case 3:
        {
            dot_kernel3<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata);
            break;
        }
        case 4:
        {
            dot_kernel4<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata);
            break;
        }
        case 5:
        {
            dot_kernel5<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata);
            break;
        }
        case 6:
        {
            dot_kernel6<blockSize><<< dimGrid, dimBlock, shMemSize >>>(d_idata1,
                    d_idata2, d_odata); 
            break;
        }
        case 7:
        {
            dot_kernel7<blockSize><<< dimGrid, dimBlock, shMemSize >>>(d_idata1,
                    d_idata2, d_odata, size); 
            break;
        }
        case 8:
        {
            dot_kernel8<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata, size); 
            break;
        }
        case 9:
        {
            dot_kernel9<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata, size); 
            break;
        }
        case 10:
        {
            dot_kernel10<<< dimGrid, dimBlock, shMemSize >>>(d_idata1, d_idata2,
                    d_odata, size); 
            break;
        }
    }
}

void gpu_dot_switch_threads(int kernel, int size, int threads, int blocks,
        float *d_idata1, float *d_idata2, float *d_odata) 
{
    dim3 dimBlock(threads,1,1);
    dim3 dimGrid (blocks, 1,1);
    int shMemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 
        threads * sizeof(float);

    switch (threads)
    {
        case 512:
            call_kernel_dot<512>(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case 256:
            call_kernel_dot<256>(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case 128:
            call_kernel_dot<128>(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case 64:
            call_kernel_dot<64 >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case 32:
            call_kernel_dot<32 >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case 16:
            call_kernel_dot<16 >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case  8:
            call_kernel_dot<8  >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case  4:
            call_kernel_dot<4  >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case  2:
            call_kernel_dot<2  >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        case  1:
            call_kernel_dot<1  >(kernel, dimGrid, dimBlock, shMemSize,
                    d_idata1, d_idata2, d_odata, size);
            break;
        default:
            printf("invalid number of threads, exiting...\n");
            exit(1);
	}
}

int CPU_dot_product(float *data1, float *data2, int num_elem, float &cpu_time)
{
    int result = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cpu_time = 0;
    float this_it;

    for(int j = 0; j < NUM_ITERATIONS; j++)
    {
        cudaEventRecord(start);
        result = dotCPU(data1, data2, num_elem);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&this_it, start, stop);
        cpu_time += this_it;
    }

    cpu_time /= NUM_ITERATIONS;
    return result;
}

/*Returns true if the kernel should be repeated afterwards*/
bool calculate_blocks_and_threads(int kernel, int invocation, int n,
        int &blocks, int &threads)
{
    switch (kernel)
    {
        case 1:
        case 2:
        case 3:
            {
                threads = (n < maxThreads) ? (n) : maxThreads;
                blocks = (n + (threads - 1)) / threads;
                return blocks != 1;
            }
        case 4:
        case 5:
        case 6:
            {
                threads = (n < maxThreads*2) ? (n/2) : maxThreads;
                blocks = (n + (threads * 2 - 1)) / (threads * 2);
                return blocks != 1;
            }
        case 7:
            {
                threads = (n < maxThreads*2) ? (n/2) : maxThreads;
                blocks = (n + (threads * 2 - 1)) / (threads * 2);
                blocks = MY_MIN(maxBlocks, blocks);
                return blocks != 1;
            }
        case 8:
            {
                if (invocation == 0)
                {
                    threads = 512;
                    blocks = (n + (threads - 1)) / threads;
                    blocks = MY_MIN(512, blocks);
                    return blocks != 1;
                }
                else
                {
                    threads = 512;
                    blocks = 1;
                    return false;
                }
            }

        case 9:
        case 10:
            {
                threads = 512;
                blocks = (n + (threads - 1)) / threads;
                return false; 
            }
        default:
            {
                printf("invalid kernel number, exiting...\n");
                exit(1);
            }
    }
}

void GPU_dot_product(int kernel, float* d_idata1, float* d_idata2,
        float *d_ONES, float *d_odata, int n, float &gpu_time)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    gpu_time = 0;

    for(int i = 0; i < NUM_ITERATIONS; i++)
    {
        /* The following line is only needed for kernels 8, 9, 10.
           It is only needed because we are reusing the output array
           for multiple test cases*/
        cudaMemsetAsync(d_odata,0,sizeof(int));
        int numThreads, numBlocks;
        int iteration_n = n;
        bool should_repeat = calculate_blocks_and_threads(kernel, 0,
                iteration_n, numBlocks, numThreads);

        cudaEventRecord(start);
        gpu_dot_switch_threads(kernel, n, numThreads, numBlocks, d_idata1,
                d_idata2, d_odata);

        // sum partial block sums for each block on GPU
        while(should_repeat)
        {
            iteration_n = numBlocks;
            should_repeat = calculate_blocks_and_threads(kernel, 1,
                    iteration_n, numBlocks, numThreads);	
            gpu_dot_switch_threads(kernel, iteration_n, numThreads, numBlocks, d_odata,
                    d_ONES, d_odata);
        }			

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_it;
        cudaEventElapsedTime(&this_it, start, stop);
        gpu_time += this_it;
    }

    gpu_time /= NUM_ITERATIONS;
}

// copy final sum/max from device to host
float transfer_result(float *d_odata, float &gpu_time)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float result = 0;

    cudaEventRecord(start);
    cudaMemcpy(&result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    return result;
}

//--- runs reductions for all array sizes -----
//--- for both CPU and GPU                -----
    template<int kernel>
void run_arrays()
{ 
    unsigned int array_size = M32 * sizeof(float);

    int maxNumBlocks = MY_MIN(M32 / maxThreads, MAX_BLOCK_SIZE);

    // allocate memory on the host
    float *h_idata1 = (float*) malloc(array_size);
    float *h_idata2 = (float*) malloc(array_size);
    float *h_ONES = (float*) malloc(array_size);
    float *h_odata = (float*) malloc(maxNumBlocks*2*sizeof(float));

    // allocate memory on the device
    float *d_idata1 = NULL;
    float *d_idata2 = NULL;
    float *d_ONES = NULL;
    float *d_odata = NULL;
    cudaMalloc((void**) &d_idata1, array_size);
    cudaMalloc((void**) &d_idata2, array_size);
    cudaMalloc((void**) &d_ONES, array_size);
    cudaMalloc((void**) &d_odata, maxNumBlocks*2*sizeof(float));

    if (!h_idata1 || !h_idata2 || !h_odata || !d_idata1 || !d_idata2 || !d_odata)
    {
        printf("Cannot allocate memory\n");
        exit(1);
    }

    srand(17);

    // create random input data on CPU
    for(int i = 0; i < M32; i++)
    { 
        h_idata1[i] = (float)(rand() % 3) - 1;
        h_idata2[i] = (float)(rand() % 3) - 1;
        h_ONES[i] = 1;
    }

    // initialize result array
    for(int i = 0; i < maxNumBlocks*2; i++)
    {
        h_odata[i] = 0.0;
    }

    //timing measurements
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy data to device memory	
    cudaEventRecord(start);
    cudaMemcpy(d_idata1, h_idata1, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idata2, h_idata2, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ONES, h_ONES, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_odata, h_odata, maxNumBlocks*2*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;	
    cudaEventElapsedTime(&transfer_in, start, stop);

    printf("\n=============\n");
    printf("kernel = %d \n", kernel);
    printf("Size GPU_dotprod CPU_dotprod CPU_time(ms) GPU_time(ms) "
            "TransferIn(ms) TransferOut(ms) Speedup_noTrf Speedup\n");

    // iterate over array sizes 2M, 8M, 32M (as required)   
    for (int i = M2; i <= M32; i*=4)
    {
        float GPUtime=0;
        float CPUtime=0;
        float transfer_out=0;
        float CPU_result=0;
        float GPU_result=0;

        GPU_dot_product(kernel, d_idata1, d_idata2, d_ONES, d_odata, i,
                GPUtime);
        GPU_result = transfer_result(d_odata, transfer_out);
        CPU_result = CPU_dot_product(h_idata1, h_idata2, i, CPUtime);

        (i == M2)?printf("%4.4s ", "2M"):((i==M32)?printf("%4.4s ", "32M"):
                printf("%4.4s ", "8M"));
        printf("%11.2f ", GPU_result);
        printf("%11.2f ", CPU_result);
        printf("%12.6f ", CPUtime);
        printf("%12.6f ", GPUtime);
        printf("%14.6f ", transfer_in);
        printf("%15.6f ", transfer_out);
        printf("%13.2f ", CPUtime/GPUtime);
        printf("%7.2f\n", CPUtime / (GPUtime+transfer_in+transfer_out));
    }
    printf("\n");

    cudaFree(d_idata1);
    cudaFree(d_idata2);    
    cudaFree(d_odata);    
    free(h_idata1);
    free(h_idata2);	
    free(h_odata);	
}

int main(int argc, char** argv)
{
    run_arrays<1>();
    run_arrays<2>();
    run_arrays<3>();
    run_arrays<4>();
    run_arrays<5>();
    run_arrays<6>();
    run_arrays<7>();
    run_arrays<8>();
    run_arrays<9>();
    run_arrays<10>();
    cudaThreadExit();
    return 0;
}
