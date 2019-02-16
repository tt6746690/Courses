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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hash.h"
#include "time_util.h"


static int hash_table_size;
static int keys_per_bucket;
static int operations_count;
static int write_percentage;

static void print_usage(char *const argv[])
{
	printf("usage: %s <hash table size> <keys per bucket> <operations count> <write percentage>\n", argv[0]);
}

static bool parse_args(int argc, char *const argv[])
{
	if (argc < 5) return false;

	hash_table_size  = atoi(argv[1]);
	keys_per_bucket  = atoi(argv[2]);
	operations_count = atoi(argv[3]);
	write_percentage = atoi(argv[4]);

	return (hash_table_size > 0) && (keys_per_bucket > 0) && (operations_count > 0) &&
	       (write_percentage > 0) && (write_percentage <= 100);
}


static bool mixed_read_write_test(hash_table_t *hash, int shift)
{
	assert(hash != NULL);
	assert(shift != 0);

	int max_key = hash_table_size * keys_per_bucket + 1;
	bool result = true;

	for (int i = 0; i < operations_count; i++) {
		bool write = (random() % 100 < write_percentage);
		int key = (int)(random() % max_key);

		if (write) {
			if (hash_put(hash, key, key + shift) != 0) {
				fprintf(stderr, "hash_put(%d, %d) failed\n", key, key + shift);
				return false;
			}
		} else {
			int value = hash_get(hash, key);
			if ((value != key + shift) && (value != -1)) {// -1 is valid since the key might not be in the table
				fprintf(stderr, "hash_get(%d) returned %d, expected %d or -1\n", key, value, key + shift);
				result = false;
			}
		}
	}
	return result;
}


static double do_perf_test(bool (*f)(hash_table_t*, int))
{
	hash_table_t *hash = hash_create(next_prime(hash_table_size));
	if (hash == NULL) {
		fprintf(stderr, "Failed to create the hash table\n");
		exit(1);
	}
	int shift = random() + 1;

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	bool success = f(hash, shift);

	clock_gettime(CLOCK_MONOTONIC, &end);	

	hash_destroy(hash);
	if (!success) exit(1);
	return timespec_to_msec(difftimespec(end, start));
}


int main(int argc, char *argv[])
{
	if (!parse_args(argc, argv)) {
		print_usage(argv);
		return 1;
	}

	srandom(time(NULL));
	printf("%f\n", do_perf_test(mixed_read_write_test));
	return 0;
}
