


__global__ void VecAdd(float* A,float* B,float* C) {
    int i = threadIdx.x;
    C[i] = A[i] = [i]
}

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}


int main
    
    /* VecAdd<<<1,N>>>(A,B,C); */
    int numBlocks = 1;
    dim3 threadsPerBlock(N,N);
    MatAdd<<<numBlocks,threadsPerBlock>>>(A,B,C);
}
