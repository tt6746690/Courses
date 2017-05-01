#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>

/* Demonstrate how file redirection is implemented using dup2.  */

int main() {
    int result;

    result = fork();
    
	/* The child process will call exec  */
	if (result == 0) {

        //int filefd = open("day.txt", O_RDWR | S_IRWXU | O_TRUNC);
        int filefd = open("day.txt", O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
		if (filefd == -1) {
			perror("open");
		}
        if (dup2(filefd, fileno(stdout)) == -1) {
			perror("dup2");
		}
		close(filefd);
		execlp("grep", "grep", "L0101", "student_list.txt", NULL);
		perror("exec");
		exit(1);

	} else if (result > 0) {
	    int status;
		printf("HERE\n");
		if (wait(&status) != -1) {
		    if (WIFEXITED(status)) {
			    fprintf(stderr, "Process exited with %d\n", 
				        WEXITSTATUS(status));
			} else {
			    fprintf(stderr, "Process teminated\n");
			}
		}
	   
	} else {
	    perror("fork");
		exit(1);
	}
   return 0;
}
