#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>

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
       a -> b
       a -> c
       a -> d
   */
	for(i = 0; i < num_kids; i++) {
		n = fork();
		if(n < 0) {
			perror("fork");
			exit(1);
        } else if(n == 0){      // check for child processes and terminates loop
            return 0;
        }
 		printf("pid = %d, ppid = %d, i = %d\n", getpid(), getppid(), i);
	}

	return 0;
}
