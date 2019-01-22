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

#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <stdbool.h>

extern int opt_ncourses;
extern int opt_ngrades;
extern const char *opt_path;

// Parse data generator arguments and store their values into the opt_* variables
bool parse_args(int argc, char *const argv[]);

#endif// _OPTIONS_H_
