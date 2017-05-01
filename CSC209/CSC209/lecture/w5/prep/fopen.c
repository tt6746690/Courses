#include <stdio.h>

int main() {
    int error;
    FILE *scores_file;

    scores_file = fopen("top10.txt", "r");
    if (scores_file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    printf("File opened: we can use it here\n");

    error = fclose(scores_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed\n");
        return 1;
    }

    return 0;
}