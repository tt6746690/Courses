#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

/*
   possible order: 
        A (BCE) (BDE)
        A (BDE) (BCE)
*/

int main() {
	int ret;

	printf("A\n");
	ret = fork();

	printf("B\n");          // printed more than once 
	if(ret < 0) {
		perror("fork");
		exit(1);

	} else if(ret == 0) {
		printf("C\n");

	} else {
		printf("D\n");
	}

	printf("E\n");          // printed more than once 
	return 0;
}
