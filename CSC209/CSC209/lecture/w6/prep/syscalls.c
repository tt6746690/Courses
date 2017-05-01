#include <stdio.h>
#include <strings.h>

int main() {
    char *str = "Hello World";

    // This should print "The length of Hello World is 11":
    printf("The length of %s is %lu\n", str, strlen(str));
    
    return 0;
}