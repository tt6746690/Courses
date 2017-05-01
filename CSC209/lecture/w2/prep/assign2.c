#include <stdio.h>

int main() {
    int x, y, z;

    x = (4 < 5);
    y = (5 < 4);
    z = ((2 < 3) || (5 < 4));

    printf("x: %d. y: %d. z: %d\n", x, y, z);
    
    return 0;
}