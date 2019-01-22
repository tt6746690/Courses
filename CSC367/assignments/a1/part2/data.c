// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Maryam Dehnavi, Bogdan Simion, Alexey Khrabrov
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Maryam Dehnavi, Bogdan Simion
// -------------

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"


__attribute__((unused))
static grade_record *load_grades(const char *path, int grades_count)
{
	assert(path != NULL);
	assert(grades_count > 0);

	FILE *f = fopen(path, "r");
	if (f == NULL) {
		perror(path);
		return NULL;
	}

	grade_record *grades = (grade_record*)malloc(grades_count * sizeof(grade_record));
	if (grades == NULL) {
		perror("malloc");
		fclose(f);
		return NULL;
	}

	if (fread(grades, sizeof(*grades), grades_count, f) < grades_count) {
		fprintf(stderr, "Invalid input file %s\n", path);
		free(grades);
		fclose(f);
		return NULL;
	}

	fclose(f);
	return grades;
}

__attribute__((unused))
static int store_grades(const char *path, grade_record *grades, int grades_count)
{
	assert(path != NULL);
	assert(grades != NULL);
	assert(grades_count > 0);

	FILE *f = fopen(path, "w");
	if (f == NULL) {
		perror(path);
		return -1;
	}

	if (fwrite(grades, sizeof(*grades), grades_count, f) < grades_count) {
		fprintf(stderr, "Failed to write to %s\n", path);
		fclose(f);
		return -1;
	}

	fclose(f);
	return 0;
}


__attribute__((unused))
static int load_courses(const char *path, course_record **courses, int *courses_count)
{
	assert(path != NULL);
	assert(courses != NULL);
	assert(courses_count != NULL);
	*courses = NULL;
	*courses_count = 0;

	FILE *f = fopen(path, "r");
	if (f == NULL) {
		perror(path);
		return -1;
	}

	if (fread(courses_count, sizeof(*courses_count), 1, f) < 1) {
		fprintf(stderr, "Invalid input file %s\n", path);
		goto error;
	}

	if ((*courses = (course_record*)malloc(*courses_count * sizeof(**courses))) == NULL) {
		perror("malloc");
		goto error;
	}

	if (fread(*courses, sizeof(**courses), *courses_count, f) < *courses_count) {
		fprintf(stderr, "Invalid input file %s\n", path);
		goto error;
	}
	fclose(f);
	return 0;

error:
	fclose(f);
	if (*courses != NULL) free(*courses);
	*courses = NULL;
	return -1;
}

__attribute__((unused))
static int store_courses(const char *path, const course_record *courses, int courses_count)
{
	assert(path != NULL);
	assert(courses != NULL);
	assert(courses_count > 0);

	FILE *f = fopen(path, "w");
	if (f == NULL) {
		perror(path);
		return -1;
	}

	if (fwrite(&courses_count, sizeof(courses_count), 1, f) < 1) {
		fprintf(stderr, "Failed to write to %s\n", path);
		fclose(f);
		return -1;
	}

	if (fwrite(courses, sizeof(*courses), courses_count, f) < courses_count) {
		fprintf(stderr, "Failed to write to %s\n", path);
		fclose(f);
		return -1;
	}
	fclose(f);
	return 0;
}


// Allocates storage for the courses array (and all the grades array that it points to); the courses array must be
// freed using the free_data() function (see below)
int load_data(const char *path, course_record **courses, int *courses_count)
{
#if EMBED_DATA
	(void)path;
	assert(courses != NULL);
	assert(courses_count != NULL);
	*courses = _courses;
	*courses_count = _courses_count;
	return 0;
#else

	if (load_courses(path, courses, courses_count) < 0) return -1;

	for (int i = 0; i < *courses_count; i++) {
		char grades_path[PATH_MAX] = "";
		snprintf(grades_path, sizeof(grades_path), "%s_%d", path, i);
		if (((*courses)[i].grades = load_grades(grades_path, (*courses)[i].grades_count)) == NULL) {
			for (int j = 0; j < i; j++) {
				free((*courses)[j].grades);
			}
			free(*courses);
			*courses = 0;
			return -1;
		}
	}
	return 0;

#endif
}

int store_data(const char *path, const course_record *courses, int courses_count)
{
	if (store_courses(path, courses, courses_count) < 0) return -1;

	for (int i = 0; i < courses_count; i++) {
		char grades_path[PATH_MAX] = "";
		snprintf(grades_path, sizeof(grades_path), "%s_%d", path, i);
		if (store_grades(grades_path, courses[i].grades, courses[i].grades_count) < 0) return -1;
	}
	return 0;
}

void free_data(course_record *courses, int courses_count)
{
#if EMBED_DATA
	(void)courses;
	(void)courses_count;
#else

	assert(courses != NULL);
	assert(courses_count > 0);
	for (int i = 0; i < courses_count; i++) {
		assert(courses[i].grades != NULL);
		free(courses[i].grades);
	}
	free(courses);

#endif
}
