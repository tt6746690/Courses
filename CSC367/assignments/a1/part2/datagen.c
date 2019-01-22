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

#include "data.h"
#include "options.h"


// Generates a data file

int main(int argc, char *argv[])
{
	if (!parse_args(argc, argv)) return 1;

	course_record *courses = (course_record*)calloc(opt_ncourses, sizeof(course_record));
	if (courses == NULL) {
		perror("calloc");
		return 1;
	}

	srandom(time(NULL));
	int result = 1;

	for (int i = 0; i < opt_ncourses; i++) {
		snprintf(courses[i].name, sizeof(courses[i].name), "CSC%ld", 100 + random() % 400);
		if ((courses[i].grades = (grade_record*)malloc(opt_ngrades * sizeof(grade_record))) == NULL) goto end;
		courses[i].grades_count = opt_ngrades;

		int base  = 50 + (random() % 40);
		int range = 10 + (random() % 20);
		for (int j = 0; j < opt_ngrades; j++) {
			courses[i].grades[j] = (grade_record){ j, (int)(base + (random() % range)) % 101 };
		}
	}

	result = (store_data(opt_path, courses, opt_ncourses) == 0) ? 0 : 1;

end:
	for (int i = 0; i < opt_ncourses; i++) {
		if (courses[i].grades != NULL) free(courses[i].grades);
	}
	free(courses);
	return result;
}
