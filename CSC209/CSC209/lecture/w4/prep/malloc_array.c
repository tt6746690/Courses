#include <stdio.h>
#include <stdlib.h>

/* 
 * Return an array of the squares from 1 to max_val.
 */
int *squares(int max_val) {
    int *result = malloc(sizeof(int) * max_val);	// pointer result is a variable 
    							// on stack, as usual for a 
							// functinon variable, 
							// the address it points to 
							// are memory on heap
    int i;
    for (i = 1; i <= max_val; i++) {
        result[i - 1] = i * i ;
    }
    return result;
}


int main() {

    int *squares_to_10 = squares(10);  // creates pointer to heap memory
    				       // the pointer itself is on stack 
				       // but the address pt points to is on heap
    
    // let's print them out 
    int i;
    for (i = 0; i < 10; i++) {
        printf("%d\t", squares_to_10[i]);
    }
    printf("\n");

    return 0;
}
