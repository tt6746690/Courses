

## Cuda by Example https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-


### chapter 5: threads

+ why use threads
    + get over constraint of arbitrary length vector


```
shared[threadIdx.x][threadIdx.y] = 1;

// Need a synchronization point between 
//      - writes to shared memory 
//      - reads from shared memory
__syncthreads();

ptr[offset] = shared[15-threadIdx.x][15-threadIdx.y];
```


### chapter 6: constant memory and events

+ bottleneck is memory bandwidth instead of arithmetic throughput
+ constant memory (64kb)
    + `__constant__ Sphere s[SPHERES];`
    + memory that does not change, read-only
    - save memory read bandwidth from global memory
        - single read can broadcast to "nearby" threads, saving up to 15 threads   
            + warp ...
        - constant memory is cached, consecutive reads to same address will not incur traffic
            + by the virtue of read-only
    - implementation notes
        - size needs to be known at compile time
        - no need for cuMalloc()
        - use cudaMemcpyToSymbol() to copy CPU memory to GPU constant memory
+ warp
    + a collection of 32 threads that are "woven together" and get executed in lockstep (at same time)
    + they execute same instruction on different data
    + constant memory access
        + broadcast a single memory read to each half-warp (16 threads)
        + i.e. if every threads in half-warp request data from same address in constant memory, GPU will generate only 1 read request, and subsequently broadcast the data to each threads
        + conclusion: 1/16 traffic (vs global memory)
    + caveats of broadcasting
        + slow performance if all threads reads different memory address
        + if access constant mem, the 16 read requests is serialized, 16x slower to place read requests
        + if access global mem, the 16 read requests are parallelized, so will be faster
+ raytracer using constant memory
    + each thread access the first `__constant__`'s sphere memory, saves lots of traffic!
    + 50% faster compared to without constant memory
+ events
    + create events then record the event
    ```
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    // work on GPU
    kernel<<<grids,threads>>>(args);

    cudaEventRecord(stop,0);
    // synchronize the event
    //      i.e. block further instructions until GPU reached `stop` event
    cudaEventSynchronize(stop);

    // safe to read from `stop`
    //      in milliseconds
    cudaEventElapsedTime(&elapsedTime, start, stop)

    // destroy 
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    ```
    + only for GPU profiling, not for a mixture of host/device code


## Chapter 7: texture memory

+ texture memory
    + read-only, cached on chip
    + designed for graphics applications, where memory access exhibit __spactial localilty__ (a thread likely to read from an address near the address that nearby threads read)
+ heat transfer