
#include "common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
            imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );

            // idea:
            //      N<=threadsPerBlock,     blocksPerGrid=1
            //      N=threadsPerBlock+1,    blocksPerGrid=2


__global__ void dot( float *a, float *b, float *c ) {

    // shared memory
    //      - private copy in each block
    //      - threads has low latent access for shared memory inside their block
    //      - threads cannot access/write shared memory of other blocks
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // cache index specific to one block
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();


    // dot product's parallel reduction
    //      - each thread keeps a partial sum inside a shared memory
    //      - do parallel reduction in O(log(threadsPerBlock)
    // 
    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        // thread synchronization
        //      in general needs synchronization between reads and writes 
        __syncthreads();
        i /= 2;
    }

    // problems with following code 
    //      **thread divergence**: some threads execute an instruction while others don't
    //      __syncthreads(): EVERY threads in the block has to execute 
    //          __syncthreads() to advance; however, if __syncthreads() is in a divergent 
    //          branch, some threads will NEVER reach __syncthreads() and hardware waits forever
    // if (cacheIndex < i) {
    //     cache[cacheIndex] += cache[cacheIndex + i];
    //    __syncthreads();
    // }

    // each block has a single number in `cache[0]`
    //      which is sum of products the threads in the block computed
    //      need to write to global memory `c`
    if (cacheIndex == 0)  // only 1 thread needs to write 1 number to global mem
        c[blockIdx.x] = cache[0];

    // left with `c[i]` contains sum produced by each `i`-th block
    // return to CPU, since GPU is inefficient at computing last few steps of reduction
}


int main( void ) {
    float   *a, *b, c, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the cpu side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N*sizeof(float) );
    cudaMalloc( (void**)&dev_b, N*sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float) );

    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice );

    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,
                                            dev_partial_c );

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(partial_c,dev_partial_c,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost );

    // finish up on the CPU side
    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf( "Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float)(N - 1) ) );

    // free memory on the gpu side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_partial_c );

    // free memory on the cpu side
    free( a );
    free( b );
    free( partial_c );
}
