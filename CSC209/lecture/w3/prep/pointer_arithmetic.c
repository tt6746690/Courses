#include <stdio.h>

int main() {

    // Try running this code in the visualizer and then try playing with the types
    // using chars instead of ints.
    
    int A[3] = {13, 55, 20};
    int *p = A;

    // dereference p
    printf("%d\n", *p);		// 13, the first element in array 

    // When we add an integer to a pointer, the result is a pointer.
    int *s;
    s = p + 1;

    // dereference s to see where this new pointer points
    printf("%d\n", *s);		// 55, the second element inthe array 

    // first add 1 to p and dereference the result
    printf("%d\n", *(p+1)); 	// address of p plus size of 1 integer = 4 byte
    				// (p+1) holds address of next element in the array 55

    // we can use array syntax on p
    printf("%d\n", p[0]);		// 13; first element in A 
    printf("%d\n", p[1]);		// 55; second element in A 

      

    p = p + 1;    
    printf("%d\n", *p);			// 55, now p points to second element in array 
    printf("%d\n", p[0]);		// 55  first element in array starting at p
    printf("%d\n", p[1]);		// 20  third element in original array 
    printf("%d\n", *(p - 1));		// 13 13 13 13 13 13 13 13 13 13 13 13 13 first element in array 

    return 0;

}
