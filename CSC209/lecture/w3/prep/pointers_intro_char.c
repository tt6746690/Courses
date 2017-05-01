#include <stdio.h>

int main() {

    char ch = 'Y';

    // ch_pt is declared as a pointer to a char
    char *ch_pt;

    // store the address of ch in the variable ch_pt
    ch_pt = &ch;

    // dereference ch_pt to print the value pointed at
    // by ch_pt 
    printf("ch_pt points to %c\n", *ch_pt);
    
    return 0;
}  //ch_pt points to Y
