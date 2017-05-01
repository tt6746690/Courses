#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

#define MAXSIZE 4096
void handle_child1(int *fd);
void handle_child2(int *fd);

/* A program to illustrate the need for select.
 *
 * The parent forks two children with a pipe to read from each of them and then
 * reads first from child 1 followed by a read from child 2.
*/
int main() {
    char line[MAXSIZE];
    int pipe_child1[2], pipe_child2[2];

    // Before we fork, create a pipe for child 1 
    if (pipe(pipe_child1) == -1) {
        perror("pipe");
    }

    int r = fork();
    if (r < 0) {
        perror("fork");
        exit(1);
    } else if (r == 0) {
        handle_child1(pipe_child1);
        exit(0);
    } else {
        // This is the parent. Fork another child, 
        // but first close the write file descriptor to child 1
        close(pipe_child1[1]);
        // and make a pipe for the second child
        if (pipe(pipe_child2) == -1) {
            perror("pipe");
        }
        // Now fork the second child
        r = fork();
        if (r < 0) {
            perror("fork");
            exit(1);
        } else if (r == 0) {
            close(pipe_child1[0]);  // still open in parent and inherited
            handle_child2(pipe_child2);
            exit(0);
        } else {
            close(pipe_child2[1]);
    
            // This is now the parent with 2 children -- each with a pipe
            //  from which the parent can read.

            // Read first from child 1
            if ((r = read(pipe_child1[0], line, MAXSIZE)) < 0) {
                perror("read");
            } else if (r == 0) {
                printf("pipe from child 1 is closed\n");
            } else {
                printf("Read %s from child 1\n", line);
            }

            // Now read from child 2
            if ((r = read(pipe_child2[0], line, MAXSIZE)) < 0) {
                perror("read");
            } else if (r == 0) {
                printf("pipe from child 2 is closed\n");
            } else {
                printf("Read %s from child 2\n", line);
            } 
        }
        // We could close all the pipes but since program is ending, we will just let
        // them be closed automatically.
    }
    return 0;
}

void handle_child1(int *fd) {
        close(fd[0]);  // we are only writing from child to parent
        printf("[%d] child\n", getpid());
        // Child will write to parent

        // Uncommenting the following while loop will show how child2's written
        // message can be *blocked* by child1:
        
        // while (1) {
        //     // do something
        // } 
        

        char message[10] = "HELLO DAD";
        write(fd[1], message, 10); 
        close(fd[1]);
}
void handle_child2(int *fd) {
        close(fd[0]);  // we are only writing from child to parent
        printf("[%d] child\n", getpid());
        // Child will write to parent
        char message[10] = "Hi mom";

        // This written message will never be processed (read by the parent) 
        // if child1 blocks:
        write(fd[1], message, 10);

        printf("[%d] child is finished writing\n", getpid());
        close(fd[1]);
}
