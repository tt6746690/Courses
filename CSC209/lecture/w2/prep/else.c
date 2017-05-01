#include <stdio.h>

int main() {
    int age = 5;

    if (age >= 13 && age <= 18) {
        printf("She's a teenager.");
    } else if (age < 13) {
        printf("She's too young.");
    } else {
        printf("She's too old.");
    }

    return 0;
}