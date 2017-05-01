#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <limits.h>

#define BUFSIZE 1024

/* This program takes two command line arguments: a number NUM and a 
 * filename FILE. It prints NUM lines from the file FILE.
 * This version has error checking.
*/

int main(int argc, char **argv) {
    FILE *fp;
    char buf[BUFSIZE];
    int i;
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s NUM FILE\n", argv[0]);
        exit(1);
    }
    
    long numlines = strtol(argv[1], NULL, 0);
    
    if (numlines <= 0) {
        fprintf(stderr, "ERROR: number of lines should be positive.");
        exit(1);
    }
    
    if ((fp = fopen(argv[2], "r")) == NULL) {
        perror("fopen");
        exit(1);
    }
    
    for (i = 0; i < numlines; i++)  {
        if ((fgets(buf, BUFSIZE, fp)) == NULL) {
            fprintf(stderr, "ERROR: not enough lines in the file\n");
            exit(1);
        }

	while (strchr(buf, '\n') == NULL) {
        if ((fgets(buf, BUFSIZE, fp)) == NULL) {
            fprintf(stderr, "ERROR: not enough lines in the file\n");
            exit(1);
        }

        printf("%s", buf);
	}

        printf("%s", buf);
    }
    
    return 0;
}