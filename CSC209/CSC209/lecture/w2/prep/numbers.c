#include <stdio.h>

int main() {
    double num1, num2;
    printf("Type two numbers separated by a space: ");
    scanf("%lf %lf", &num1, &num2);
    printf("You typed %f and %f\n", num1, num2);
    return 0;
}
