/* File:     Naive matrix multiplication
* Adapted from: http://www.cs.usfca.edu/~peter/math202/blocked.c
 *
 * Purpose:  Run a standard matrix multiply
 *
 * Compile:  gcc -g -Wall [-DDEBUG] -o matmul matmul.c
 * Run:      ./matmul <order of matrices>
 *              <-> required argument, [-] optional argument
 *
 * Output:   Elapsed time for standard matrix multiply
 *           If the DEBUG flag is set, the product matrix as
 *           computed by each method is also output.
 *
 * Notes:
 * 1.  The file timer.h should be in the directory containing
 *     the source file.
 * 3.  Set the DEBUG flag to see the product matrices
 * 5.  There are a number of optimizations that can be made to 
 *     the source code that will improve the performance of both
 *     algorithms.
 * 6.  Note that unless the DEBUG flag is set the product matrices will, 
 *     in general, be different using the two algorithms, since the two 
 *     algorithms use identical storage for A and B, but they assume 
 *     the storage has different structures.
 * 8.  This has received *very* little testing.  Students who find
 *     and correct bugs will receive many gold stars.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset

#define PAD 504

struct record {
	double value;
	char padding[PAD];
};

// Global Variables 
const double DRAND_MAX = RAND_MAX;
struct record *A, *B, *C;
struct record *C_p;
int n, b;
int n_bar, b_sqr;

void Usage(char prog_name[]);
void Get_matrices(struct record A[], struct record B[], int n);
void Mat_mult(void);
void Print_matrix(struct record C[], int n);

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
	volatile char MARKER_START, MARKER_END;
	/* Record marker addresses */
	FILE* marker_fp = fopen("marker","w");
	if(marker_fp == NULL ) {
		perror("Couldn't open marker file:");
		exit(1);
	}
	fprintf(marker_fp, "%p %p", &MARKER_START, &MARKER_END );
	fclose(marker_fp);

	MARKER_START = 33;


   if (argc != 2) Usage(argv[0]);
   n = strtol(argv[1], NULL, 10);

   A = malloc(n*n*sizeof(struct record));
   B = malloc(n*n*sizeof(struct record));
   C = malloc(n*n*sizeof(struct record));
   if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "Can't allocate storage!\n");
      exit(-1);
   }

   Get_matrices(A, B, n);

   Mat_mult();
#  ifdef DEBUG 
   printf("Standard algorithm\n");
   Print_matrix(C, n);
#  endif


   free(A);
   free(B);
   free(C);
	MARKER_END = 34;
   return 0;
}  /* main */

/*-------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message showing how the program is used and quit
 * In arg:    prog_name:  the program name
 */
void Usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <order of matrices> \n",
         prog_name);
   exit(0);
}  /* Usage */


/*-------------------------------------------------------------------
 * Function:  Get_matrices
 * Purpose:   Read in the factor matrices from stdin or generate
 *            them if argc == 3
 * In args:   n:  order of the matrices
 * Out args:  A, B: the matrices
 */
void Get_matrices(struct record A[], struct record B[], int n ) {
   int i;

      for (i = 0; i < n*n; i++) {
         A[i].value = random()/DRAND_MAX;
         B[i].value = random()/DRAND_MAX;
      }
}  /* Get_matrices */


/*-------------------------------------------------------------------
 * Function:    Mat_mult
 * Purpose:     Use the standard algorithm for matrix multiplication
 * Globals in:  A, B:  factor matrices
 *              n:     order of matrices
 * Globals out: C:  the product matrix
 */
void Mat_mult(void) {
   int i, j, k;

   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
         C[i*n + j].value = 0.0;
         for (k = 0; k < n; k++)
            C[i*n + j].value += A[i*n + k].value * B[k*n + j].value;
      }
   }
}  /* Mat_mult */


/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print a matrix on stdout
 * In args:   n:  order of matrix
 *            A:  the matrix
 */
void Print_matrix(struct record C[], int n) {
   int i, j;

   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         printf("%.2e ", C[i*n+j].value);
      printf("\n");
   }
}  /* Print_matrix */

