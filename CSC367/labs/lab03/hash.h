// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion, Maryam Dehnavi, Alexey Khrabrov
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
// -------------

#ifndef _HASH_H_
#define _HASH_H_

struct _hash_table_t;
typedef struct _hash_table_t hash_table_t;

// Get the smallest prime number that is not less than n (for hash table size computation)
int next_prime(int n);

// Create a hash table with 'size' buckets; the storage is allocated dynamically using malloc(); returns NULL on error
hash_table_t *hash_create(int size);
// Release all memory used by the hash table, its buckets and entries
void hash_destroy(hash_table_t *table);

// Valid keys and values are >= 0

// Returns -1 if key is not found
int hash_get(hash_table_t *table, int key);
// Returns 0 on success, -1 on failure
int hash_put(hash_table_t *table, int key, int value);

#endif// _HASH_H_
