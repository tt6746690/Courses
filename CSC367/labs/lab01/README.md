
2. Requirements
You can find the starter code on scinet teach under /home/t/teachcsc367/CSC367Starter/labs/lab1/starter_code.tgz so copy this into your repository and make sure you can do your first commit. Please make sure to read carefully over the code, including the licenses and the instructions from the comments.

The starter code provided starts with an M x N matrix (M rows, N columns), each column of which represents an N-dimensional vector, and computes the average length of these vectors. By vector length we mean Euclidean norm in the N-dimensional space, not the number of components.

Your task is to measure the execution time (in milliseconds, not including data generation), profile the code, and implement an optimized version (in lab1-opt.c).

Important:

Do not change the output format given in the starter code.
You may *not* modify the declaration of the matrix, nor the macros NDIMS / NVECS.
Timing the code
For this part, you must inspect the man pages for the clock_gettime function, and use clock_gettime to measure the time it takes to run the avg_vec_len function. You must use the CLOCK_MONOTONIC clock. Your time must be reported in milliseconds.

We have provided a set of helper functions in time_util.h, but it is your responsibility to add timing correctly to your code, to measure the appropriate piece of code.

Hint: The difftimespec function takes two timespec structures and calculates the elapsed time. Keep in mind to report the time in ms.

Profiling the code
For this part, you will familiarize yourselves with two important tools: gprof and perf. The former will be useful in determining where most of the time is spent, and the latter you will be using to look at some architectural counters, to figure out your program's cache locality.

 

gprof: The most basic usage involves simply adding the "-pg" flag to the compilation and linking of your program, and then running your code. After you run your code, a "gmon.out" file will be generated. To inspect the gprof details, simply use:


 $ gprof ./myprogram gmon.out > gprof_analysis.txt
You can then inspect the gprof_analysis.txt file for details on how much time is spent in each function. For further details, please refer to the full documentation for gprof  (see here (![Links to an external site.](https://sourceware.org/binutils/docs/gprof/)). Discuss with your TA any confusion you might have with respect to the code you are profiling.
 

perf: Please refer to the man pages for perf. Specifically, you might want to look at the man page for perf-stat. You must average your measurements over at least 5 runs (use the perf built-in option).

Hints: 
1. Consider measuring relevant performance counters, for example: L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses, cache-misses, cache-references, etc. Feel free to consult the following tutorial (Links to an external site.)Links to an external site. as well. If certain Perf versions can only measure up to a few architectural counters at a time, you can capture the counters in multiple (separate) perf runs. Keep in mind that some counters may not be supported in all processors.

You might want to write a script for running the profiling, to simplify your task and automate gathering measurements.
Once you're clear on what the performance bottleneck is, optimize the code and place the optimized code in the opt file, according to the instructions in the comments (see the TODO comments).

3. Submission
Ensure that you have a repository directory created by MarkUs for this exercise. In this directory, add and commit+push all the files required to compile and run your code (all the source code and the Makefile). You shouldn't need to add any new files to the starter code, but if you do, make sure that you update the Makefile accordingly. Make sure your code compiles and runs correctly on Scinet.