#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	int i; 
	pid_t result;

	i = 5;
	printf("%d\n", i); 

	result = fork();

	if (result > 0) {
		i = i + 2; 		// parent process
	} else if (result == 0) {
		i = i - 2;		// child process
	} else {
		perror("fork");
	}

	printf("%d\n", i);
	return 0;
}