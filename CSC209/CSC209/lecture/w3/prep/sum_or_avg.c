#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

    // Ensure that the program was called correctly.
    if (argc < 3) {
        printf("Usage: sum_or_avg <operation s/a> <args ...>");
        return 1;	// NON ZERO EXIT CODE -- TERMINATE ABNORMALLy
    }

    int total = 0;
    
    int i;
    for (i = 2; i < argc; i++) {	// start at 2 to exclude excutable name + flag
        total += strtol(argv[i], NULL, 10);
    }

    if (argv[1][0] == 'a') {	// note argv is an array of pointer to char
	    			// argv[1] is the pointer to the second char array - string 
				// argv[1][0] is the first char in first string 
				
        // We need to cast total to float before the division.
        // Try removing the cast and see what happens.
        double average = (float) total / (argc - 2);	// casting is required because operands are a bunch of integers, fraction may be possible 
        printf("average: %f \n", average);
    } else {
        printf("sum: %d\n", total);
    }

    return 0;
}
