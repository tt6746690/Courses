#include <stdio.h>
#include <stdlib.h>

int main() {

    // this allocates space for the 2 pointers
    int **pointers = malloc(sizeof(int *) * 2); 
    // the first pointer points to a single integer
    pointers[0] = malloc(sizeof(int));
    // the second pointer pointes to an array of 3 integers
    pointers[1] = malloc(sizeof(int) * 3);

    // let's set their values
    *pointers[0] = 55;

    pointers[1][0] = 100;
    pointers[1][1] = 200;
    pointers[1][2] = 300;

    printf("[0][0] = %d", pointers[0][0]);

    
    // do other stuff with this memory

    // now time to free the memory as we are finished with the data-structure
    // first we need to free the inner pieces
    free(pointers[0]);
    free(pointers[1]);
    // now we can free the space to hold the array of pointers themselves
    free(pointers);
    
    return 0;
}
