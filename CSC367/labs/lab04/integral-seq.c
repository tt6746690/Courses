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
#include <omp.h>
#include <stdio.h>


static double f(double x)
{
	return sin(sin(x));
}

static const double step = 0.0001;

static double integral_f(double a, double b)
{
	assert(a < b);

	double s = 0.0;
	for (double x = a; x < b; x += step) {
		s += step * f(x);
	}
	return s;
}


int main()
{
	//TODO: measure time (in milliseconds) taken to execute primes_count()
	double time_msec = 0.0;
	time_msec = omp_get_wtime();
	double result = integral_f(0.0, 100.0);
	time_msec = (omp_get_wtime() - time_msec) * 1000.;

	printf("%f\n", result);
	printf("%f\n", time_msec);
	return 0;
}
