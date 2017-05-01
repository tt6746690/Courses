#include <stdio.h>


int main(){
	// cant use pointer because memory must be allocated for scanf to store 
	char name1[128];
	char name2[128];

	while(scanf("%s %s", name1, name2)){
		printf("%s %s", name2, name1);
	}
	return 0;
}

