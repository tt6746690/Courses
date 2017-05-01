#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int main(int argc, char **argv) {

    int i;
    int n;
    int num_kids;

    if(argc != 2) {
        fprintf(stderr, "Usage: forkloop <numkids>\n");
        exit(1);
    }

    num_kids = strtol(argv[1], NULL, 10);

    /* 
        a -> b -> c -> d
    */
    for(i = 0; i < num_kids; i++) {
        n = fork();
        printf("pid = %d, ppid = %d, i = %d\n", getpid(), getppid(), i);

        if(n < 0) {
            perror("fork");
            exit(1);
        } else if(n > 0){       // terminates parent process, wait for childs before retun
            pid_t pid;
            int status;
            if( (pid = wait(&status)) == -1 ) {
                perror("wait");
            }        
            return 0;
        }
    }


    return 0;
}
