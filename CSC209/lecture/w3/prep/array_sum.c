#include <stdio.h>

int sum(int *A, int size) {

    // arrays are not pass by value, only pointer to first element in array is possed to the function 
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += A[i];
    }
    return total;
}


int main() {

    int scores[4] = {4, 5, -1, 12};
    printf("total is %d\n", sum(scores, 4));

    int ages[3] = {10, 12, 19};
    printf("total is %d\n", sum(ages, 3));

    return 0;
}
