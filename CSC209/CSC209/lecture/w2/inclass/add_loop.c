#include <stdio.h>

#define SIZE 1000

int main() {
    int ind, amt;
    // You can initialize an array in C by giving a value to 
    // the the first element or elements.  The rest of the values
    // will be initialized to 0.
    int a[SIZE]= {4};
    
    int result;
    /*
    // You can play around with the input you type in to see what scanf will
    // return.  This is currently commented out because it isn't part of the 
    // main question.
    result = scanf("%d %d", &ind, &amt);
    printf("got: %d %d result = %d\n", ind, amt, result);
    result = scanf("%d %d", &ind, &amt);
    printf("got: %d %d result = %d\n", ind, amt, result);
    */
    while(scanf("%d %d", &ind, &amt) != EOF) {
        a[ind] += amt;
    }

    int i;
    for(i = 0; i < SIZE; i++) {
        printf("Value at %d is %d\n", i, a[i]);
    }
    
    
    return 0;
}