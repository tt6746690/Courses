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
#include <pthread.h>
#include <stdio.h>

#include "data.h"
#include "time_util.h"

//TODO: parallelize the code


// Compute the historic average grade for a given course. Updates the average value in the record
void compute_average(course_record *course)
{
	assert(course != NULL);
	assert(course->grades != NULL);

	course->average = 0.0;
	for (int i = 0; i < course->grades_count; i++) {
		course->average += course->grades[i].grade;
	}
	course->average /= course->grades_count;
}

// Compute the historic average grades for all the courses
void compute_averages(course_record *courses, int courses_count)
{
	assert(courses != NULL);

	for (int i = 0; i < courses_count; i++) {
		compute_average(&(courses[i]));
	}
}


int main(int argc, char *argv[])
{
	course_record *courses;
	int courses_count;
	// Load data from file; "part2data" is the default file path if not specified
	if (load_data((argc > 1) ? argv[1] : "part2data", &courses, &courses_count) < 0) return 1;

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	compute_averages(courses, courses_count);
	clock_gettime(CLOCK_MONOTONIC, &end);

	for (int i = 0; i < courses_count; i++) {
		printf("%s: %f\n", courses[i].name, courses[i].average);
	}

	printf("%f\n", timespec_to_msec(difftimespec(end, start)));

	free_data(courses, courses_count);
	return 0;
}
