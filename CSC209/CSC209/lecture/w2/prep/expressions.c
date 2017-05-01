#include <stdio.h>

int main() {
    int x, y;

    x = 2;
    y = (x + 2) * (x + 5);

    x = 10;     // the value of y does not change
                // until after the *next* line has
                // executed!

    y = (x + 2) * (x + 5);

    return 0;
}
