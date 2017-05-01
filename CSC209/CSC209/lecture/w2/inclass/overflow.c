#include <stdio.h>

// Example illustrating array overflow in the extreme
// Probably when you run this, you will get a Segmentation Fault error
// which means that your program is accessing memory that it doesn't have
// permission for.

int main() {
    int a[4];
    int i;
    for(i = 0; i < 10000; i++) {
        a[i] = i;
        printf("a[%d] = %d\n", i , a[i]);
    }
    for(i = 0; i < 10000; i++) {
        printf("a[%d] = %d\n", i , a[i]);
    }
    return 0;
}