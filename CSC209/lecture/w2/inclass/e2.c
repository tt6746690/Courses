#include <stdio.h>

#define SIZE 10


int main(){
	int ind, amt;
	int a[SIZE];
	int i;
	for (i=0; i< SIZE; i++){
		a[i] = 0;
	}
	// int a[SIZE] = {0}; alternative way of initializing

	while(scanf("%d %d", &ind, &amt) != EOF){
		a[ind] += amt;
		printf("array[%d]=%d\n", ind, a[ind]);
	}

	for(i = 0; i< SIZE; i++){
		printf("value at %d is %d", i, a[i]);
	}
  
	return(0);
}
