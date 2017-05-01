#include <stdio.h>

#define LINE_LENGTH 80

/* Reads and prints the contents of a file (top10.txt). */
int main() {
    FILE *scores_file;
    int error;
    char line[LINE_LENGTH + 1];  // +1 for the null-terminator
  
    scores_file = fopen("top10.txt", "r");

    // Check if scores_file was opened properly:
    if (scores_file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // If fgets fails to read any characters, it returns NULL.
    // We can use this fact to tell if we've reached the end
    // of the file:
    while (fgets(line, LINE_LENGTH + 1, scores_file) != NULL) {
        printf("%s", line);
    }
  
    error = fclose(scores_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed\n");
        return 1;
    }

    return 0;
}