/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#include "time_util.h"
#include "lab2helper.h"

float *input;
float *weights;
float *result;
int32_t n;
int32_t threads;

//Implement the next 4 functions.

// Sequential implementation of the array "input" with 
// the array "weights", writing the result to "results".
void scale_sequential()
{
    for (int i = 0; i < n; ++i) {
        result[i] = weights[i] * input[i];
    }
}

// Parallel sharded implementation of the array "input" with 
// the array "weights", writing the result to "results".
void* scale_parallel_sharded(void *val)
{
    /* Remember that a thread needs an id. 
     * You should treat val as a pointer
     * to an integer representing the id of the thread.
     */
    int chunksize = n / threads;
    int chunkoffset = (*(int*)val) * chunksize;
    for (int i = chunkoffset; i < chunksize + chunkoffset; ++i) {
        result[i] = weights[i] * input[i];
    }
    return NULL;
}

// Parallel strided implementation of the array "input" with 
// the array "weights", writing the result to "results".
void* scale_parallel_strided(void *val)
{
    /* Remember that a thread needs an id. 
     * You should treat val as a pointer
     * to an integer representing the id of the thread.
     */
    int stride = threads;
    for (int i = *(int*)val; i < n; i += stride) {
        result[i] = weights[i] * input[i];
    } 
    return NULL;
}

enum{
    SHARDED,
    STRIDED
};

void start_parallel(int mode)
{
    /* Create your threads here
     * with the correct function as argument (based on mode).
     * Notice that each thread will need an ID as argument,
     * so that the thread can tell which indices of the array it should
     * work on. For example, to create ONE thread on sharded mode,
     * you would use:
     *   int id = 0;
     *   pthread_t worker;
     *   pthread_create(&worker, NULL, scale_parallel_sharded, &id);
     *
     * You want to create "thread" threads, so you probably need
    * an array of ids and an array of pthread_t.
     * Don't forget to wait for all the threads to finish before
     * returning from this function (hint: look at pthread_join()).
     */

    pthread_t tpool[threads];
    int idx[threads];
    int rc;

    for (int i = 0; i < threads; ++i) {
        idx[i] = i;
        if (mode == SHARDED) {
            rc = pthread_create(&tpool[i], NULL, scale_parallel_sharded, (void*)&idx[i]);
        } 
        if (mode == STRIDED) {
            rc = pthread_create(&tpool[i], NULL, scale_parallel_strided, (void*)&idx[i]);
        }

        if (rc) {
            fprintf(stderr, "return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (int i = 0; i < threads; ++i) {
        rc = pthread_join(tpool[i], NULL);
        if (rc) exit(-1);
    }

}


int main(int argc, char **argv)
{
    /**************** Don't change this code **********************/
    if (argc != 2)
    {
        printf("Usage: %s num_threads\n", argv[0]);
        exit(1);
    }

    threads = atoi(argv[1]);
    n = (1<<29); // 512M floats =  4 x 512MB = 2GB.
    input = gen_input_from_memory(n, 0);
    weights = gen_weight_from_memory(n, 0);
    result = malloc(n * sizeof(float));
    // Total = 3 x 2GB = 6GB.


    /**************** Change the code below **********************/
    // You must keep the order in which implementations are called.
    // You must keep the call to reset_array in between invocations.
    {
        //call your sequential function here
        //and time it. Do not include the lines
        //below in the measurement.
        float time = 0;
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        scale_sequential();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = timespec_to_sec(difftimespec(end, start));
        printf("sequential = %10.8f\n", time);
        dump_output_to_file(result, n, "sequential_output.txt");
        reset_array(result, n);
    }
    {
        //call your sequential function here *FOR A SECOND TIME*
        //and time it. Do not include the lines
        //below in the measurement.
        //This may seem weird, but observe what happens.
        float time = 0;
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        scale_sequential();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = timespec_to_sec(difftimespec(end, start));
        printf("sequential = %10.8f\n", time);
        dump_output_to_file(result, n, "sequential_output.txt");
        reset_array(result, n);
    }
    {
        //call your start_parallel function here on strided mode
        //and time it. Do not include the lines
        //below in the measurement.
        float time = 0;
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_parallel(STRIDED);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = timespec_to_sec(difftimespec(end, start));
        printf("parallel strided = %10.8f\n", time);
        dump_output_to_file(result, n, "strided_output.txt");
        reset_array(result, n);
    }
    {
        float time = 0;
        //call your start_parallel function here on sharded mode
        //and time it. Do not include the lines
        //below in the measurement.
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_parallel(SHARDED);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = timespec_to_sec(difftimespec(end, start));
        printf("parallel sharded = %10.8f\n", time);
        dump_output_to_file(result, n, "sharded_output.txt");
        reset_array(result, n);
    }
}
