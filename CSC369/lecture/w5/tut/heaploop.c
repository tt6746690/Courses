#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define RECORD_SIZE  128 

struct krec {
	double d[RECORD_SIZE];
};

void heap_loop(int iters) {
	int i;
	struct krec *ptr = malloc(iters * sizeof(struct krec));

        /* initializes first element of array */
	for(i = 0; i < iters; i++) {
		ptr[i].d[0] = (double)i;
	}
	free(ptr);
}

void stack_loop(int iters) {
	int i;
	struct krec a[iters];
	for(i = 0; i < iters; i++) {
		a[i].d[0] = (double)i;
	}
}

int main(int argc, char ** argv) {
	/* 
         * Note there maybe init code run before main().
         * Markers used to bound trace regions of interest 
         * so only region on stack between Markers should be of interest
         */
	volatile char MARKER_START, MARKER_END;
	/* Record marker addresses */
	FILE* marker_fp = fopen("./marker","w");
	if(marker_fp == NULL ) {
		perror("Couldn't open marker file:");
		exit(1);
	}
	fprintf(marker_fp, "%p %p", &MARKER_START, &MARKER_END );
	fclose(marker_fp);

	MARKER_START = 33;

        /* total of 500 * 8 (sizeof double) * 128 (sizeof krec) = 512000 bytes  
         * Assuming pages are 4096 bytes, then 
         * Number of pages for storage = 512000/4096 = 125, and
         * each page holds 4 krec struct
         * So every page will be hit 4 times, 
         */
	heap_loop(500);
	MARKER_END = 34;

	return 0;
}
