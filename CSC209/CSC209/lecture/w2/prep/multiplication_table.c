#include <stdio.h>

/* A small program that prints the contents of a 10 * 10
   multiplication table using nested loops. */
int main() {
    int row, col;

    for (row = 1; row < 10; row++) {
        for (col = 1; col < 10; col++) {
            // '\t' is the tab character:
            printf("%d\t", col * row);
        }

        // '\n' is the newline character:
        printf("\n");

    }

    return 0;
}