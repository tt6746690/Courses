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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int comm_size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size = comm_size;

	int *row = (int*)malloc(size * sizeof(int));
	int *trans = (int*)malloc(size * size * sizeof(int));
	if ((row == NULL) || (trans == NULL)) {
		perror("malloc");
		MPI_Finalize();
		exit(1);
	}

	// Generate this node's row of the matrix
	for (int i = 0; i < size; i++) {
		row[i] = (rank + 1) + (i + 1) * 1000;
	}

	// Print the original matrix
	for (int i = 0; i < comm_size; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (i == rank) {
			for (int j = 0; j < size; j++) {
				printf("%d ", row[j]);
			}
			printf("\n");
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//TODO: measure time (in milliseconds) taken to execute matrix transposition
	//TODO: perform matrix transposition using Gather/Allgather
	double time_msec = 0.0, elapsed = 0.0;
	time_msec = MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		if (rank == i) {
			MPI_gather(row, size, MPI_INT, trans + i*size, size, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	if (rank == 0) {
		for (int i = 0; i < size; ++i) {
			row[i] = trans[i*size + rank];
		}
	}

	elapsed = MPI_Wtime() - time_msec;
	printf("process %d time elapsed: %f ms\n", rank, elapsed * 1000);

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		// Print the transposed matrix
		printf("\n");
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				printf("%d ", trans[i * size + j]);
			}
			printf("\n");
		}
		// Print execution time
		printf("%f\n", time_msec);
	}

	free(row);
	free(trans);
	MPI_Finalize();
	return 0;
}
