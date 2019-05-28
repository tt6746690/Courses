

// single block
__global__ void VecAdd(float* A,float* B,float* C) {
    int i = threadIdx.x;
    C[i] = A[i] = [i]
}

// single block
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

// multiple blocks
__global__ void MatAddMultiBlock(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}


int  () {
    /* VecAdd<<<1,N>>>(A,B,C); */
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
    MatAdd<<<numBlocks,threadsPerBlock>>>(A,B,C);
}
