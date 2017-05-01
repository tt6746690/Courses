#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char **argv) {
    // Call malloc with a size that's too large to allocate
    // memory for:
    char *name = malloc(LONG_MAX);

    if (name == NULL) {
    	perror("malloc");  // prints an error message to stderr
    	exit(1);
    }

    return 0;
}