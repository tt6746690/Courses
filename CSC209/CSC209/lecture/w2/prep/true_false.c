#include <stdio.h>

int main() {

    // 0 is false so not printed
    if (0) {
      printf("first\n");
    }

    // 1 is true so printed 
    if (1) {
      printf("second\n");
    }

    // Every non-zero value is interpreted as true, so
    // the printf statement in this condition will execute:
    if (2) {
      printf("third\n");
    }
    return 0;
}
