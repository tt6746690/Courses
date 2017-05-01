#include <stdio.h>
#include <string.h>

int main() {
    char s1[30] = "University of C Programming";
    char *p;

    // find the index of the first 'v'
    p = strchr(s1, 'v');
    if (p == NULL) {
        printf("Character not found\n");
    } else {
        printf("Character found at index %ld\n", p - s1);	// prints 3
    }

    // find the first token (up to the first space)
    p = strchr(s1, ' ');
    if (p != NULL) {
        *p = '\0';
    }
    printf("%s\n", s1);

    return 0;
}
