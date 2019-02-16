/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Do not modify anything in this file!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#include <string.h>

float* gen_input_from_memory(int size, int seed) {
  float *target = malloc(size * sizeof(float));

  for (int i = 0; i < size; ++i)
    target[i] = i + seed;

  return target;
}

float* gen_weight_from_memory(int size, int seed) {
  return gen_input_from_memory(size, seed);
}

int dump_output_to_file(float *output, int size, char *filename)
{
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("could not open output file\n");
        return -1;
    }

    int i;
    float signature = 0;
    for(i = 0; i < size; i++)
    {
      signature += output[i];
    }

    fprintf(f, "%f\n", signature);
    fclose(f);
    return 0;
}

void reset_array(float *array, int size) {
  memset(array, 0, size * sizeof(float));
}
