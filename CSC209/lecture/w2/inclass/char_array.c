#include <stdio.h>

/* This example illustrates array overflow using character arrays.
 */

int main() {
    char course_prefix[3] = {'c', 's', 'c'};
    char other[3] = {'b', 'c', 'b'};
                
    int i;
    for(i = 0; i < 24; i++) {
        course_prefix[i] = 'z';
    }
    printf("%s\n", course_prefix);
    printf("%s\n", other);
                                                      
    return 0;
}
