#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFSIZE 256

/* This program takes two command line arguments: a number NUM and a 
 * filename FILE. It prints NUM lines from the file FILE.
 * This version has no error checking.
*/

int main(int argc, char **argv) {
    FILE *fp;
    char buf[BUFSIZE];
    int i;
    
    long numlines = strtol(argv[1], NULL, 0);
    
    fp = fopen(argv[2], "r");
    
    for (i = 0; i < numlines; i++)  {
        fgets(buf, BUFSIZE, fp);
        printf("%s", buf);
    }
    
    return 0;
}