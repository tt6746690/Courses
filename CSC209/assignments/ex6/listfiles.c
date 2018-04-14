#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>

/* This program takes a single argument that is a path to a directory.
 * The program stores the contents of the directory in an array, calls
 * qsort to sort them in descending alphabetical order (from z to a) and
 * then prints the array.

 * Most of the code is given for you.  You are asked to complete a few sections.
 * The purpose of this exercise is to give you some practice with function
 * pointers, and some practice using readdir.  You will probably need to read
 * the man page for qsort and the man page for readdir to complete this
 * program.

 * A Makefile and a test directory have been provided for you as an example
 * of how you might test your program.
 */

#define MAXDIRS 100

/* The compare function that will be passed into qsort to sort
 * an array of dirents by name.
 */
int compare_names(const void *n1, const void *n2) {
    // return 0 for same and non zero value for different 
    // descending order then add -1
    return -strcmp(((struct dirent *) n1)->d_name, ((struct dirent *) n2)->d_name); 
}

/* Print an array of dirents by name
 */
void show_dirs(struct dirent *dp, int size) {
	for(int i = 0; i < size; i++) {
		printf("%s\n", dp[i].d_name);
	}
}

int main(int argc, char **argv) {
	/* We have to make a design decision here. It is costly in terms of
	 * time to read through a directory twice just to count the number of
	 * directory entries so that we know the number of entries to create
	 * an array of the appropriate size. We can also make a guess that
	 * we will never use this program on a directory with more entries than
	 * some maximum, knowing that we will have to recompile the program if
	 * we do hit our limit.
	 */
	struct dirent dirs[MAXDIRS];
	struct dirent *dp;
	
	if(argc != 2) {
		fprintf(stderr, "Usage: listdir PATH\n");
		exit(1);
	}
	
	DIR *dirp = opendir(argv[1]);
	if(dirp == NULL) {
		perror("opendir");
		exit(1);
	}
    /* Store all directory entries that do not begin with '.' in dirs
	 * If the number of entries exceeds MAXDIRS, exit the loop and 
	 * print an error message.
	 * 
	 * Note that the value of i will be the number of elements in the array
	 * (the number of entries in the directory that do not begin with '.')
	 */
	 int i = 0;

	 // TODO: Add the code to populate the array with directory entries.
	 // Tip: "man readdir"
     while ((dp = readdir(dirp)) != NULL){
         // append dirent to array iff d_name != "." and ".."
            i++;
     }

	if(i == MAXDIRS && dp != NULL) {
		fprintf(stderr, 
		    "Error: program does not support more than %d entries\n", MAXDIRS);
	}
	closedir(dirp);
	
	
	// Sort dirs by name in descending alphabetical order
	// TODO: call qsort with the appropriate arguments to sort the array of
	// struct dirents
    qsort(dirs, i, sizeof(struct dirent), compare_names);

	printf("List the directory entries sorted in descending order:\n");
	show_dirs(dirs, i);
	
	return 0;
}
