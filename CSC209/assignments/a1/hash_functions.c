#include <stdio.h>

void hash(char *hash_val, long block_size) {

    // initialize empty hash_val array
    for (int i = 0; i< block_size; i++){
        hash_val[i] = '\0';  // \0 is NUL with ASCII = 0
    }

    char in_str[1];
    char *hash_val_ptr = hash_val;
    // input character to create hash
    while (scanf("%c", in_str) != EOF){

        // XOR hash function
        *hash_val_ptr = *hash_val_ptr ^ *in_str;
        hash_val_ptr++;

        // reset index
        if(hash_val_ptr == (hash_val + block_size)){
            // printf("address of hash_val=%p address of ptr=%p", hash_val, hash_val_ptr);
            hash_val_ptr = hash_val;
        }
    }
}

int check_hash(const char *hash1, const char *hash2, long block_size) {
    // should not use  string function here because  
    //  We are dealing with binary representations mostly for hashes 

    for (int i = 0; i< block_size; i++){
        if (hash1[i] != hash2[i]){
            return i;
        }
    }
    return block_size;
}

