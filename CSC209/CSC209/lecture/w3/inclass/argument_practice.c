#include <stdio.h>

int main(int argc, char **argv){
	printf("we have %d command-line arguments.\n", argc- 1);
	printf("The two arguments are %s and %s", argv[1], argv[2]);

	return 0;
}
/*
 *If no argument is given to cml
 *
 * we have 0 command-line arguments.
 * The two arguments are (null) and TERM_SESSION_ID=w0t0p2:14911A36-F452-4381-88CD-2A5F93B9B4AF%
 */
