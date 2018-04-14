/* this solution needs error checking! */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

/* Read a user id and password from standard input, 
   - create a new process to run the validate program
   - send 'validate' the user id and password on a pipe, 
   - print a message 
        "Password verified" if the user id and password matched, 
        "Invalid password", or 
        "No such user"
     depending on the return value of 'validate'.
*/

/* Use the exact messages defined below in your program." */

#define VERIFIED "Password verified\n"
#define BAD_USER "No such user\n"
#define BAD_PASSWORD "Invalid password\n"
#define OTHER "Error validating password\n"


int main(void) {
    char userid[10];
    char password[10];

    /* Read a user id and password from stdin */
    printf("User id:\n");
    scanf("%s", userid);
    printf("Password:\n");
    scanf("%s", password);

    int fd[2];
    // open pipe for transfering data from parent to child process 
    if(pipe(fd) == -1){
        perror("pipe");
        exit(1);
    }

    // call fork after pipe 
    int result = fork();
    if (result == 0){       // child process 
        // child not writing to pipe 
        if(close(fd[1]) == -1){
            perror("close");
            exit(1);
        }

        /* fprintf(stderr, "fd[0] = [%d]; stdin = [%d]\n", fd[0], fileno(stdin)); */
        // redirect read end of pipe to stdin
        if(dup2(fd[0], fileno(stdin)) == -1){
            perror("dup2");
            exit(1);
        }
        /* fprintf(stderr, "fd[0] = [%d]; stdin = [%d]\n", fd[0], fileno(stdin)); */
        if(close(fd[0]) == -1){
            perror("close");
            exit(1);
        }

        // child process call validate 
        execl("./validate", "validate", NULL);
        // should not reach here
        perror("exec");
        exit(1);

    } else if(result > 0){  // parent process
        //  parent not reading from pipe
        if(close(fd[0]) == -1){
            perror("close");
            exit(1);
        }
        
        // write userid and password to pipe
        if(write(fd[1], userid, 10) == -1 ){
            perror("write");
            exit(1);
        }
        if(write(fd[1], password, 10) == -1 ){
            perror("write");
            exit(1);
        }

        // close write end of pipe 
        if(close(fd[1]) == -1){
            perror("close");
            exit(1);
        }

        // wait for child process to terminate
        int status;
        if(wait(&status) == -1){
            perror("wait");
            exit(1);
        } else {
            if(!WIFEXITED(status)){
                fprintf(stderr, OTHER);
            } else {
                if( WEXITSTATUS(status) == 0 ) {
                    fprintf(stderr, VERIFIED);
                } else if( WEXITSTATUS(status) == 2 ) {
                    fprintf(stderr, BAD_PASSWORD);
                } else if( WEXITSTATUS(status) == 3 ) {
                    fprintf(stderr, BAD_USER);
                }
            }
            
        }

    } else {
        perror("fork");
        exit(1);
    }

	

    return 0;
}
