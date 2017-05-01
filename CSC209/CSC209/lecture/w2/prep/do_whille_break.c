#include <stdio.h>

int main() {
    int sum = 0;
    int curr_int;

    do {
        printf("Enter an integer: ");
        scanf("%d", &curr_int);
        if (curr_int <= 0) {
            break;
        }
        sum += curr_int;
    } while (1); // A FLAG for break... this is infinity loop

    printf("The total is %d\n", sum);

    return 0;
}
