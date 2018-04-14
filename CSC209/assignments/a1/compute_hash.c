#include <stdio.h>
#include <stdlib.h>


// Hash manipulation functions in hash_functions.c
void hash(char *hash_val, long block_size);
int check_hash(const char *hash1, const char *hash2, long block_size);

#ifndef MAX_BLOCK_SIZE
    #define MAX_BLOCK_SIZE 1024
#endif

/* Converts hexstr, a string of hexadecimal digits, into hash_val, an an
 * array of char.  Each pair of digits in hexstr is converted to its
 * numeric 8-bit value and stored in an element of hash_val.
 *    hash_val = [ 2 hex digits, 2 hex digits = 8-bit = 1 char, ... ]
 * Preconditions:
 *    - hash_val must have enough space to store block_size elements
 *    - hexstr must be block_size * 2 characters in length
 */

void xstr_to_hash(char *hash_val, char *hexstr, int block_size) {
    for(int i = 0; i < block_size*2; i += 2) {
        char str[3];
        str[0] = hexstr[i];
        str[1] = hexstr[i + 1];
        str[2] = '\0';
        hash_val[i/2] = strtol(str, NULL, 16);
    }
}

// Print the values of hash_val in hex
void show_hash(char *hash_val, long block_size) {
    for(int i = 0; i < block_size; i++) {
        printf("%.2hhx ", hash_val[i]);
    }
    printf("\n");
}

/**
 * takes in
 *      1. block_size, which is the number of bytes that computed hash should be
 *      2. A string of hex digits (each 1 bit) representing a hash value (optional)
 *          where hash value is the value returned by a hash function
 */
int main(int argc, char **argv) {

    if ( argc < 2 || argc > 3){
      printf("Usage: compute_hash BLOCK_SIZE [ COMPARISON_HASH ]");
      return 1;
    }

    long block_size = strtol(argv[1], NULL, 10);

    if ( block_size <= 0 || block_size >= MAX_BLOCK_SIZE){
      printf(" The block size should be a positive integer less than %d.", MAX_BLOCK_SIZE);
      return 1;
    }

    // hash from stdin
    char hash_val[block_size];
    hash(hash_val, block_size);
    
    printf("hash val from stdin is \n");
    show_hash(hash_val, block_size);

    if ( argc == 3 ){
        char comparison_hash[block_size];
        xstr_to_hash(comparison_hash, argv[2], block_size);
        printf("comparison hash is \n");
        show_hash(comparison_hash, block_size);
        int result = check_hash(hash_val, comparison_hash, block_size);
        printf("Compare hash result: %d", result);
    }

    /*int res1 = check_hash("skskd", "skskd", 5);        // should return 5
    int res2 = check_hash("abcdef", "abcfed", 6);		// should return 3
    printf("result 1 is %d; result 2 is %d", res1, res2);*/
    return 0;
}
