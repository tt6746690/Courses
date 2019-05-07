

## Cuda by Example https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-



### chapter 5 


+ why use threads
    + get over constraint of arbitrary length vector


```
// (x,y) determines position of thread in the grid
//      linearize (x,y) -> offset to determine unique offset
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;
```


```
// shared memory
//      - private copy in each block
//      - threads has low latent access for shared memory inside their block
//      - threads cannot access/write shared memory of other blocks
__shared__ float cache[threadsPerBlock];

// cache index specific to one block
int cacheIndex = threadIdx.x;

// thread synchronization
//      in general
//          - needs synchronization between reads and writes 
__syncthreads();

// dot product
//      - each thread keeps a partial sum inside a shared memory
//      - do parallel reduction in O(log(threadsPerBlock)

// threadsPerBlock must be powers of 2
int i = blockDim.x/2;
while (i != 0) {
    if (cacheIndex < i) {
        cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
}

// each block has a single number in `cache[0]`
//      which is sum of products the threads in the block computed
//      write to global memory 
if (cacheIndex == 0)        // only 1 thread needs to write 1 number to global mem
    c[blockIdx.x] = cache[0];

// left with `c[i]` contains sum produced by each `i`-th block
// return to CPU, since GPU is inefficient at computing last few steps of reduction

c = 0;
for (int i = 0; i < blocksPerGrid; i++) {
    c += partial_c[i];
}


// problems with following code 
//      thread divergence: some threads execute an instruction while others don't
//      problem with __syncthreads():
//          EVERY threads in the block has to execute __syncthreads() to advance
//          however, if __syncthreads() is in a divergent branch, some threads will 
//              NEVER reach __syncthreas()
//          hardware waits forever ...
if (cacheIndex < i) {
    cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
}
```