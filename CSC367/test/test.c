#include <omp.h>

void run(int*a, int size) {
    int tid = omp_get_thread_num();
    for(i = 0; i < size; i++) {
        printf("TID[%d] - a[%d] = %d\n", tid, i, a[i]);
    }
}

int main() {
    int i, size=8;
    int *a = (int*)malloc(size*sizeof(int));
    for(i = 0; i < size; i++) {
        a[i] = i+1;
    }
    #pragma omp parallel shared(size, a) num_threads(8)
    {
        run(a,size);
    }
}

