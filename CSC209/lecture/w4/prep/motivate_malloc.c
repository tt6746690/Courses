#include <stdio.h>
#include <stdlib.h>


int *set_i() {
    int *i_pt = malloc(sizeof(int));
    *i_pt = 5;
    return i_pt;

    // Try commenting out the lines above and uncommenting the ones below
    // to see teh problem if you don't use malloc.
    //int i = 5;
    //return &i;
}

// a function that appears to have nothing to do with i and pt
int some_other_function() {
    int junk = 999;
    return junk;
}

int main () {
    int *pt = set_i(); 

    // try this program with and without this function call 
    some_other_function();

    printf("but if I try to access i now via *pt I get %d\n", *pt);
    return 0;
}