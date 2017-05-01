#include <stdio.h>

int main() {
    printf("Hello, world!\n");

    int age = 50;
    printf("Here is an integer: %d\n", age);

    double temperature = 98.5;
    printf("This number %f is a floating-point number.\n", temperature);

    double one_third = 1.0 / 3.0;
    printf("Here is a double for one-third: %f\n", one_third);
    printf("Here we control the number of decimal digits: %.3f\n", one_third);

    // You can use multiple format specifiers in a printf statement:
    printf("Hello %d. Hello %f.\n", age, temperature);

    // The following printf statements will give you warnings:
    // this is not an error, the code would still run!
    printf("Hello %d. Hello %f.\n%f\n", age, temperature);
    printf("Hello %f. Hello %f.\n", age, temperature);

    return 0;
}
