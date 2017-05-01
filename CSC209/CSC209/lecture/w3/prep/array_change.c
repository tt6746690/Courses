#include <stdio.h>

int change(int *A) {
    // We explicitly change element 0 to 50 
    // in order to demonstrate that this change
    // lasts in the array that was passed to the function
    A[0] = 50;
}


int main() {

    int scores[4];
    // set the first element to 4
    scores[0] = 4;

    // call the function which is supposed to change it to 50
    change(scores);

    // What will be printed?
    // Will the function have changed the array scores or just a local
    // copy of that array?
    // ----> the array is changed globally, because array is not pass by value therefore no local copy is made in the function 
    printf("First element in array has value %d\n", scores[0]);
    // So 50 is printed here
    return 0;
}
