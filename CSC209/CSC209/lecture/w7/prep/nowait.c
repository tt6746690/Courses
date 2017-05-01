#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>

int main() {
    int result;
    int i, j;

    printf("[%d] Original process (my parent is %d)\n", 
            getpid(), getppid());

    for (i = 0; i < 5; i++) {
        result = fork();

        if (result == -1) {
            perror("fork:");
            exit(1);
        } else if (result == 0) {   //child process
            for (j = 0; j < 5; j++) {
                printf("[%d] Child %d %d\n", getpid(), i, j);
                usleep(100);
            }
            exit(i);
        }
    }

    printf("[%d] Parent about to terminate\n", getpid());
    return 0;
}
