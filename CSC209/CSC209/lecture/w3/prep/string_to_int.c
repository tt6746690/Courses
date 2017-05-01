#include <stdio.h>
#include <stdlib.h>

int main() {
    //char *s = "17";		// first character is 1 second char is 7
    char *s = "  -17";
    int i = strtol(s, NULL, 10);	// string to long 
    					//  1. s = string to be converted 
					//  2. endpointer = NULL
					//  3. base number = 10

    printf("i has the value %d\n", i);


    s = "  -17 other junk.";		
    char *leftover;
    i = strtol(s, &leftover, 10);	// note address of leftover, a pointer to char 
    					// used as input to parameters 

    printf("i has the value %d\n", i);	// i = -17
    printf("leftover has the value %s\n", leftover);	// leftover = " other junk."
    							// note trailing whitespace is not truncated
    return 0;
}
