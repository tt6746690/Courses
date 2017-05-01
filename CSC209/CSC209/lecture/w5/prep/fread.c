#include <stdio.h>

#define NUM_ELEMENTS 5

int main() {
    FILE *data_file;
    int error;
    int numbers[NUM_ELEMENTS];
    int i;    
  
    data_file = fopen("array_data", "rb");
    if (data_file == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return 1;
    }
  
    fread(numbers, sizeof(int), NUM_ELEMENTS, data_file);
  
    for (i = 0; i < NUM_ELEMENTS; i++) {
        printf("%d ", numbers[i]);
    }
    
    printf("\n");
  
    error = fclose(data_file);
    if (error != 0) {
        fprintf(stderr, "Error: fclose failed\n");
        return 1;
    }

    return 0;
}