#include <stdio.h>

int main() {
    printf("'Cause the players gonna\n");

    int i;
    for (i = 0; i < 5; i++) {
        printf("%d\n", i);
        printf("play\n");
    }

    // Check if i has the value 5 after the loop:
    printf("%d\n", i); 

    return 0;
}