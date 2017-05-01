#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char **argv){
    int fds[argc][2];           // variable length array 

    for(int i = 1; i < argc; i++){
        if( pipe(fds[i]) == -1 ){
            perror("pipe");
        }
        int result = fork();
        if (result < 0){
            perror("fork");
            exit(1);
        } else if(result == 0){     // child process
            // child only writes to pipe, so close reading ends 
            close(fds[i][0]);

            // before forked, parent had open reading ends for all previously forked children 
            // so close those to avoid potential errors 
            for(int j = 1; j < i; j ++){
                close(fds[j][0]);
            }

            // parent had open reading ends to all previously forked children, so close those
            int length = strlen(argv[i]);
            write(fds[i][1], &length, sizeof(length));

            // close write end 
            close(fds[i][1]);
            exit(0);
        } else {
            // parent, before next loop iteration, close end of pipe
            close(fds[i][1]);
        }

        // only parent gets here 
        int sum = 0;
        // read one integer from each childm print it and add to sum
        for(int i = 1; i < argc; i++){
            int num = 0;
            int num_read;
            if((num_read = read(fds[i][0], &num, sizeof(int)) == sizeof(int))){
                printf("child %d sent %d", i, num);
                sum += num;
            } else {
                fprintf(stderr, "something is wrong\n");
            }
        }
        printf("length of all args is %d\n", sum);
        return 0;


    }
}

