#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv){
	
	// note *(argv+1) is equivalent to argv[1] 
	for (int i = 0; i< strtol(argv[2], NULL, 10); i++){
		printf("Blue jay go!\n");
	}
	return 0;
}
