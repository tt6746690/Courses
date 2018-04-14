#include <stdio.h>
#include <stdlib.h>

#include "hash.h"

#define BLOCK_SIZE 8

char *hash(char *hash_val, FILE *f) {
    char ch;
    int hash_index = 0;

    for (int index = 0; index < BLOCK_SIZE; index++) {
        hash_val[index] = '\0';
    }

    while(fread(&ch, 1, 1, f) != 0) {
        hash_val[hash_index] ^= ch;
        hash_index = (hash_index + 1) % BLOCK_SIZE;
    }

    return hash_val;
}


int check_hash(const char *hash1, const char *hash2) {
    for (long i = 0; i < BLOCK_SIZE; i++) {
        if (hash1[i] != hash2[i]) {
            printf("Index %ld: %c\n", i, hash1[i]);
            return 1;
        }
    }
    return 0;
}


void show_hash(char *hash){
    printf("[");
    for (int i = 0; i < BLOCK_SIZE; i++){
        printf("%.2hhx", hash[i]);
    }
    printf("]\n");

}
