#include <stdio.h>

/* Prints dots to standard error. */
int main() {
    int i = 0;

    for (;;) {
        if (( i++ % 50000000) == 0) {
            fprintf(stderr, ".");
        }
    }

    return 0;
}