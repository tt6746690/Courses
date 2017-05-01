#include <stdio.h>

int main() {
    FILE *scores_file, *output_file;
    int error, total;
    char name[81];
  
    scores_file = fopen("top10.txt", "r");
    if (scores_file == NULL) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }
  
    output_file = fopen("names.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file\n");
        return 1;
    }
  
    while (fscanf(scores_file, "%80s %d", name, &total) == 2) {
        printf("Name: %s. Score: %d.\n", name, total);
        fprintf(output_file, "%s\n", name);
    }
  
    error = fclose(scores_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed on input file\n");
        return 1;
    }

    error = fclose(output_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed on output file\n");
        return 1;
    }

    return 0;
}