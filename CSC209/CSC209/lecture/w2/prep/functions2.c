#include <math.h>
#include <stdio.h>

int main() {
    double larger_num = fmax(2 * 8.1, 10 * 19.177);
    printf("Expressions are evaluated before being passed as arguments, so the max is %f\n\n", larger_num);

    double num1 = 1000.0;
    double num2 = 2.35;
    larger_num = fmax(num1, (num2 + 1700) / num1);
    printf("Variables, and expressions with variables, can also be used as parameters.\n");
    printf("The max value of %f and %f is: %f\n\n", num1, (num2 + 1700) / num1, larger_num);

    printf("Here's the max of %f and %f again: %f\n\n", num1, (num2 + 1700) / num1, fmax(num1, num2));

    double nested_result = fmax(num2, fmax(0.1 * num1, 10));
    printf("This number is the result of a function call that uses another function call as one of its parameters: ");
    printf("%f \n\n", nested_result);

    return 0;
}
