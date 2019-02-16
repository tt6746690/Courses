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

// So that it does asserts even with -DNDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hash.h"


static const size_t hash_size = 101;

static void put_get(hash_table_t *hash)//1
{
	hash_put(hash, 123, 321);
	int v = hash_get(hash, 123);
	assert(v == 321);
}

static void put_get_wrong_key(hash_table_t *hash)//2
{
	hash_put(hash, 123, 321);
	int v = hash_get(hash, 321);
	assert(v == -1);
}

static void get_empty_table(hash_table_t *hash)//3
{
	int v = hash_get(hash, 123);
	assert(v == -1);
}

static void collisions(hash_table_t *hash)//4
{
	srandom(time(NULL));
	int shift = random() + 1;

	int count = hash_size * 100;
	for (int i = 0; i < count; i++) {
		hash_put(hash, i, i + shift);
	}
	for (int i = 0; i < count; i++) {
		int v = hash_get(hash, i);
		assert(v == i + shift);
	}
}

//...


static void (*test_cases[])(hash_table_t*) = {
	put_get,//1
	put_get_wrong_key,//2
	get_empty_table,//3
	collisions,//4
	//...
};

int main()
{
	for (int i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
		hash_table_t *hash = hash_create(hash_size);
		assert(hash != 0);
		test_cases[i](hash);
		hash_destroy(hash);
		printf("Test case %d successful\n", i + 1);
	}

	printf("All test cases successful\n");
	return 0;
}
