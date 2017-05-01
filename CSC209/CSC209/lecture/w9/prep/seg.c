#include <stdio.h>

/* This program deliberately dereferences a NULL pointer in order
   to cause a segmentation fault. */
int main() {
    int *ptr = NULL;
    *ptr = 3;
    printf("Value at ptr is %d\n", *ptr);
    
    return 0;
}