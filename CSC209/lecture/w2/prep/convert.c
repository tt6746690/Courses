#include <stdio.h>

int main() {
    double cm, inches;

    // %lf stands for long float
    // &cm stands for location / address of cm so that it can be changed
    printf("Type a number of centimeters: ");
    scanf("%lf", &cm);

    inches = cm * 0.393701;
    printf("%lf centimeters is %lf inches.\n", cm, inches);

    return 0;
}
