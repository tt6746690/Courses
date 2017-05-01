#include <stdio.h>

/* A small program that uses scanf to read an integer. 
   We can use redirection to read from a file instead
   of from the keyboard.
 */
int main() {
    int number;
    scanf("%d", &number);
    printf("The number is %d\n", number);
    
    return 0;
}