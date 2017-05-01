#include <stdio.h>

int main() {
    // Before running this code, predict what it will print. Then
    // run it to see if your prediction matches what actually happens.

    int i = 5;
    int j = 10;

    int k = i / j;
    printf("int k is %d\n", k);	 // k = 0, fractional part is truncated  

    double half = 0.5;
    k = half;
    printf("int k is %d\n", k);  // int k = 0, if assigned integer a double   

    double d = i / j;
    printf("double d is %f\n", d);  // doub d = 0.0000000


    // i / j when evaluated has a type 
    // type of a result of an expression depends on 
    // 	1. operator 
    // 	2. type of operands 
    //in C, division of two integer is integer, independent of variable results are stored into 

    d = (double) i / j;  	// here int i is cast to double i; the resrult of expression over division is a double as long as one of the operands is a double 
    printf("double d is %f\n", d);	// d =  0.500000

    return 0;
}
