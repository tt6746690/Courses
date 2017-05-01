#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv) {

	int i;
	int n;
	int iterations;

	if(argc != 2) {
		fprintf(stderr, "Usage: forkloop <iterations>\n");
		exit(1);
	}

	iterations = strtol(argv[1], NULL, 10);

    /*
       iterations   process count   example (2^0 + 2^1 + 2^2 + 2^(iterations - 1))
       2            4               a->b | a->c, b->d
       3            8               1->2 | 1->3, 2->4 | 1->5, 3->6, 2->7, 4->8
    */
	for(i = 0; i < iterations; i++) {
		n = fork();
		if(n < 0) {
			perror("fork");
			exit(1);
		}
		//printf("pid = %d, i = %d\n", getpid(), i);
		printf("ppid = %d, pid = %d, i = %d\n", getppid(), getpid(), i);
	}

	return 0;
}
