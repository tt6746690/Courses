#include <stdio.h>
int main() {

    // Run this code in the memory visualizer and 
    // look at the addresses and values carefully.

    // variables i and j are integers
    int i = 7;
    int j = i;   // note j does not change because 7 is stored in some other addresses in memory
    // variable pt is declared as type pointer to int
    int * pt;
    // the address of i is assigned as the value of pt
    pt = &i;
    // 9 is assigned to the variable pointed-at by pt
    *pt = 9;
    printf("Value of i: %d", i); // 9 value of i changed

    // First the expression on the right is evaluated to give 10.
    // Then, 10 is assigned to the variable pointed-at by pt.
    *pt = *pt + 1;
    // note the address pointed to by pt, i.e. i, has value reassigned to 9+1 = 10
    printf("pt points to %d\n", *pt);
    return 0;
}
