#include <stdio.h>

int main() {
    float gpa = 2.3;

    // && is the "and" operator:
    if (gpa >= 0.0 && gpa <= 4.0) {
        printf("GPA is valid\n");
    } else {
        printf("GPA is not valid\n");
    }

    float gpa1 = 3.3;
    float gpa2 = 2.2;

    // || is the "or" operator:
    if (gpa1 >= 3.0 || gpa2 >= 3.0) {
        printf("One or both GPAs are at least 3.0\n");
    } else {
        printf("Both GPAs are below 3.0\n");
    }

    float gpa3 = 2.3;

    // ! is the "not" operator:
    if (!(gpa3 < 0.0 || gpa3 > 4.0)) {
        printf("GPA is valid\n");
    } else {
        printf("GPA is not valid\n");
    }
    return 0;
}