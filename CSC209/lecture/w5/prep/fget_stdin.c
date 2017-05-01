#include <stdio.h>

#define LINE_LENGTH 80

/* This program uses fgets to read from standard input (stdin)
   and prints whatever was entered as input.

   As long as standard input has not been redirected, this
   program reads strings from the keyboard.
 */
int main() {
    char line[LINE_LENGTH + 1];

    while (fgets(line, LINE_LENGTH + 1, stdin) != NULL) {
        printf("You typed: %s", line);
    }

    return 0;
}