#include <stdio.h>
#include <stdlib.h>

int play_with_memory() {
    int i;
    int *pt = malloc(sizeof(int));

    i = 15; 	// i has memory on stack 
    *pt = 49;	// pt is stored on stack, but the address 
    		// it points to are located on heap 
    
    // What happens if you comment out this call to free?
    // pt will cause memory leak
    free(pt);

    // What happens if you uncomment these statements?
    // printf("%d\n", *pt);
    // *pt = 7;
    // printf("%d\n", *pt);
	// will print: 49 and 7.
	// note that free does not reset the value of pt
	// will simply tell management system that this block of memory is free to use
    return 0;
}

int main() {
    play_with_memory();
    // if int *pt is not freed, there is no way to access the memory again
    // also the address is allocated, hence will not be used again
    play_with_memory();
    play_with_memory();
    return 0;
}
    
    
