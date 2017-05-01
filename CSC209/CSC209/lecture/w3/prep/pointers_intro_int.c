#include <stdio.h>

int main() {
    int i;
    i = 5;
    printf("Value of i: %d\n", i);      // 5
    printf("Address of i: %p\n", &i);   // O

    // declare a pointer pt that will point to an int
    int *pt;    // int * is the data type
    // set pt to hold the address of i
    pt = &i;

    // print the value of the pointer (which is the address of i)
    printf("Value of pt: %p\n", pt);
    // the pointer itself has an address, print that
    printf("Address of pt: %p\n", &pt);

    // print the value in the address that is itself stored in pt
    printf("Value pointed to by pt: %d\n", *pt);  // * is dereferencing operator here
    return 0;

    // prints to
    /*
      Value of i: 5
      Address of i: 0x7fff52223658
      Value of pt: 0x7fff52223658
      Address of pt: 0x7fff52223650
      Value pointed to by pt: 5
    */
}
