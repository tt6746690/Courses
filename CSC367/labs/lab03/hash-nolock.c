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

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "hash.h"


struct _hash_table_t {
	int size;// should be a prime number

	//TODO
};


static bool is_prime(int n)
{
	assert(n > 0);
	for (int i = 2; i <= sqrt(n); i++) {
		if (n % i == 0) return false;
	}
	return true;
}

// Get the smallest prime number that is not less than n (for hash table size computation)
int next_prime(int n)
{
	for (int i = n; ; i++) {
		if (is_prime(i)) return i;
	}
	assert(false);
	return 0;
}


// Create a hash table with 'size' buckets; the storage is allocated dynamically using malloc(); returns NULL on error
hash_table_t *hash_create(int size)
{
	assert(size > 0);

	//TODO
	return NULL;
}

// Release all memory used by the hash table, its buckets and entries
void hash_destroy(hash_table_t *table)
{
	assert(table != NULL);

	//TODO
}


// Returns -1 if key is not found
int hash_get(hash_table_t *table, int key)
{
	assert(table != NULL);

	//TODO
	return -1;
}

// Returns 0 on success, -1 on failure
int hash_put(hash_table_t *table, int key, int value)
{
	assert(table != NULL);

	//TODO
	return -1;
}
