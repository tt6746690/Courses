#include <stdio.h>
#include <stdlib.h>

#include "hash.h"
#define BLOCK_SIZE 8

char *hash(FILE *f) {

    char *hash_val = malloc(BLOCK_SIZE + 1);
    char in_str[1];
    char *hash_val_ptr = hash_val;

    // initialize empty hash_val array
    for (int i = 0; i< BLOCK_SIZE + 1; i++){
        hash_val[i] = '\0';  // \0 is NUL with ASCII = 0
    }

    // read one character at a time to create XOR hash
    while (fread(in_str, 1, 1, f) == 1){

        // XOR hash function
        *hash_val_ptr = *hash_val_ptr ^ in_str[0];
        hash_val_ptr++;

        // reset index
        if(hash_val_ptr == (hash_val + BLOCK_SIZE)){
            /* printf("address of hash_val=%p address of ptr=%p", hash_val, hash_val_ptr); */
            /* printf("; current hash = %s\n", hash_val); */
            hash_val_ptr = hash_val;
        }
    }
    return hash_val;
}
