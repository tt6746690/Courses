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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "options.h"


// Generates the C source code file for embedding data into compiled executables (see Makefile for details)

int main(int argc, char *argv[])
{
	if (!parse_args(argc, argv)) return 1;
	const char *path = "./part2_data.c";

	FILE *f = fopen(path, "w");
	if (f == NULL) {
		perror(path);
		return 1;
	}

	srandom(time(NULL));

	if (fprintf(f, "#include \"data.h\"\n") < 0) goto error;

	for (int i = 0; i < opt_ncourses; i++) {
		int base  = 50 + (random() % 40);
		int range = 10 + (random() % 20);
		if (fprintf(f, "grade_record _grades_%d[%d] = {\n", i, opt_ngrades) < 0) goto error;
		for (int j = 0; j < opt_ngrades; j++) {
			if (fprintf(f, "{%d,%d},\n", j, (int)(base + (random() % range)) % 101) < 0) goto error;
		}
		if (fprintf(f, "};\n") < 0) goto error;
	}

	if (fprintf(f, "course_record _courses[%d] = {\n", opt_ncourses) < 0) goto error;
	for (int i = 0; i < opt_ncourses; i++) {
		if (fprintf(f, "{ \"CSC%ld\", _grades_%d, %d, 0.0 },\n", 100 + random() % 400, i, opt_ngrades) < 0) goto error;
	}
	if (fprintf(f, "};\n") < 0) goto error;

	if (fprintf(f, "int _courses_count = %d;\n", opt_ncourses) < 0) goto error;

	fclose(f);
	return 0;

error:
	perror("fprintf");
	fclose(f);
	return 1;
}
