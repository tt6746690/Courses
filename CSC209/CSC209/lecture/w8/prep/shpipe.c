#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>

/* equivalent to sort < file1 | uniq */

int main() {
    int fd[2], r;

    /* Create the pipe */
    if ((pipe(fd)) == -1) {
        perror("pipe");
        exit(1);
    }

    if ((r = fork()) > 0) { // parent will run sort
        // Set up the redirection from file1 to stdin
        int filedes = open("file1", O_RDONLY);

        // Reset stdin so that when we read from stdin it comes from the file
        if ((dup2(filedes, fileno(stdin))) == -1) {
            perror("dup2");
            exit(1);
        }
        // Reset stdout so that when we write to stdout it goes to the pipe
        if ((dup2(fd[1], fileno(stdout))) == -1) {
            perror("dup2");
            exit(1);
        }

        // Parent won't be reading from pipe
        if ((close(fd[0])) == -1) {
            perror("close");
        }

        // Because writes go to stdout, noone should write to fd[1]
        if ((close(fd[1])) == -1) {
            perror("close");
        }

        // We won't be using filedes directly, so close it.
        if ((close(filedes)) == -1) {
            perror("close");
        }

        execl("/usr/bin/sort", "sort", (char *) 0);
        fprintf(stderr, "ERROR: exec should not return \n");

    } else if (r == 0) { // child will run uniq

        // Reset stdi so that it reads from the pipe
        if ((dup2(fd[0], fileno(stdin))) == -1) {
            perror("dup2");
            exit(1);
        }

        // This process will never write to the pipe.
        if ((close(fd[1])) == -1) {
            perror("close");
        }

        // SInce we rest stdin to read from the pipe, we don't need fd[0]
        if ((close(fd[0])) == -1) {
            perror("close");
        }

        execl("/usr/bin/uniq", "uniq", (char *) 0);
        fprintf(stderr, "ERROR: exec should not return \n");

    } else {
        perror("fork");
        exit(1);
    }
    return 0;
}
