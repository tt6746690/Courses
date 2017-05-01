#include <stdio.h>

/* set *largest_pt to the largest element pointed to by the array */
void find_largest(int **A, int A_size, int *largest_pt) {
	// note we use **A here because address of int *A[2] is passed to array, which is equivalent as int **A
	// Also note pointer to largest--> largest_pt is passed as address to int largest 
    *largest_pt = **A;		// assigns first element in A to largest_pt 
    for (int i = 1; i < A_size; i++) {
        if (*A[i] > *largest_pt) {
            *largest_pt = *A[i];
        }
    }
}

int main() {

    int i = 81;
    int j = -4;
    // now A holds 2 pointers
    int *A[2] = {&j, &i};		// array of pointers 

    
    int largest;

    find_largest(A, 2, &largest);	// note A really means &A[0], not pass by value.
    printf("largest is %d\n", largest);

    return 0;
}
