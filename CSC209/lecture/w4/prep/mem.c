#include <stdio.h>
#include <stdlib.h>

int sum(int a, int b){
   int *p = malloc(sizeof(int) * 500);

   return a + b;
}

int z;

int main() {
    z = sum(1, 3);

    char *c_ptr = "Hi!";
    printf("Hi!");

    // This will give you a segmentation fault:
    // int *p = NULL;
    // p[0] = 'a';

    return 0;
}
