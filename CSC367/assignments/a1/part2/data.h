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

#ifndef _DATA_H_
#define _DATA_H_


// Contains information about a single student's grade in a course
typedef struct _grade_record {
	int student;// Student ID
	int grade;// Grade (0-100)
} grade_record;

typedef struct _course_record {
	char name[8];
	grade_record *grades;// The grades of all the students that have taken this course
	int grades_count;// Length of the 'grades' array
	double average;// Historic average grade; must be (re)computed by this program
} course_record;


#if EMBED_DATA
extern course_record _courses[];
extern int _courses_count;
#endif


// The data for a set of courses is stored in multiple files: one for the course records, and a separate file for grade
// records for each course (with '_i' appended to their names, where i is the index in the array of courses)

// Allocates storage for the courses array (and all the grades array that it points to); the courses array must be
// freed using the free_data() function (see below)
int load_data(const char *path, course_record **courses, int *courses_count);

int store_data(const char *path, const course_record *courses, int courses_count);

void free_data(course_record *courses, int courses_count);


#endif// _DATA_H_
