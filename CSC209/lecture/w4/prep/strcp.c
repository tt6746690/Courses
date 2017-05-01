#include <stdio.h>
#include <string.h>

int main() {
    char s1[5];
    char s2[32] = "University of";

    // This is unsafe because s1 may not have enough space
    // to hold all the characters copied from s2.
    //strcpy(s1, s2);

    // This doesn't necessarily null-terminate s1 if there isn't space.
    strncpy(s1, s2, sizeof(s1));
    // So we explicitly terminate s1 by setting a null-terminator.
    s1[4] = '\0';

    printf("%s\n", s1);	// prints Univ  with terminating null  
    printf("%s\n", s2); // prints University of
    return 0;
}
