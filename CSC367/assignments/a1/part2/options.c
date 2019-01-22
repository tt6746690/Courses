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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#include "options.h"


int opt_ncourses;
int opt_ngrades;
const char *opt_path;

static const int max_ncourses = 8;

static void print_usage(char *const argv[])
{
	printf("usage: %s [ -c <number of courses> -g <(total) number of grades> <data file path> ]\n", argv[0]);
}

// Parse data generator arguments and store their values into the opt_* variables
bool parse_args(int argc, char *const argv[])
{
	assert(argc > 0);
	assert(argv != NULL);

	// Default values of parameters
	opt_ncourses = get_nprocs();
	opt_ngrades = 10000000;
	opt_path = "part2data";

	char option;
	while ((option = getopt(argc, argv, "c:g:")) != -1) {
		switch (option) {
			case 'c': opt_ncourses = atoi(optarg); break;
			case 'g': opt_ngrades  = atoi(optarg); break;
			default:
				print_usage(argv);
				return false;
		}
	}

	if ((opt_ncourses <= 0) || (opt_ngrades < opt_ncourses)) {
		fprintf(stderr, "Invalid arguments\n");
		print_usage(argv);
		return false;
	}

	if (opt_ncourses > max_ncourses) opt_ncourses = max_ncourses;
	opt_ngrades /= opt_ncourses;
	if (optind < argc) opt_path = argv[optind];

	return true;
}
