#include <stdio.h>

int main(void) {
    char text[20];
    text[0] = 'h';
    text[1] = 'e';
    text[2] = 'l';
    text[3] = 'l';
    text[4] = 'o';
    text[5] = '\0';

    // We can print each character in the array one at a time.
    // But we don't know what junk is in the array after the null character.
    int i;
    for (i = 0; i < 20; i++) {
        printf("%c", text[i]);
    }
    printf("\n");

    // we can use %s as a format specifier inside printf to print strings
    printf("%s\n", text);
    return 0;
}
