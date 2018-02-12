
#include <stdio.h>

double factorial(double x) {
    if (x == 1) return 1;
    else return x * factorial(x-1);
}

int main(int argv, char**argc){
    printf("factorial of 200 is %f\n", factorial(175));
}
