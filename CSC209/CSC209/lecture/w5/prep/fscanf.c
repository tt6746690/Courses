#include <stdio.h>

int main() {
    FILE *scores_file;
    int error, total;
    char name[81];
  
    scores_file = fopen("top10.txt", "r");
    if (scores_file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
  
    // Like scanf, fscanf returns the number of items successfully
    // read.
    // Here we compare the return value of fscanf to 2, since we
    // expect it to find two things on each call: a string
    // and an integer.
    while (fscanf(scores_file, "%80s %d", name, &total) == 2) {
        printf("Name: %s. Score: %d.\n", name, total);
    }
  
    error = fclose(scores_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed\n");
        return 1;
    }

    return 0;
}