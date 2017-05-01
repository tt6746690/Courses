#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 80

int main(int argc, char **argv) {
    FILE *fp;
    char s[MAX_LINE_LENGTH + 1];

    if ((fp = fopen(argv[1], "r")) == NULL) {
        perror("fopen");
        exit(1);
    }

    if (fgets(s, MAX_LINE_LENGTH, fp) == NULL) {
        fprintf(stderr, "no line could be read from the file\n");
        exit(1);
    }

    printf("One line from the file: %s", s);

    return 0;
}