#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char **argv){

    for (int i = 0; i < argc; i++){
        int r = fork();
        if (r < 0){
            perror("fork");
            exit(1);
        } else if (r == 0){
            int length = strlen(argv[i]);
            exit(length);
        }
    }
    // only parent process here now, since all child exits 
    int sum = 0;
    for (int i = 1; i < argc; i++){
        int status;
        if(wait(&status) == -1){
            perror("wait");
        } else {
            if(WIFEXITED(status)){
                sum += WEXITSTATUS(status);
            }
        }
    }
    printf("%d", sum);
    exit(1);
}


