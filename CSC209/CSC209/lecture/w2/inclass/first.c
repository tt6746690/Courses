#include <stdio.h>

// Basic scanf example

int main() {
    int num1[5];
    int num2[5];

    for(int i = 0; i < 5; i++) {

        scanf("%d%d", &num1[i], &num2[i]);
        printf("%d plus %d equals %d\n", 
                num1[i], num2[i], num1[i] + num2[i]);
    }
    return 0;
}