#include <stdlib.h>
#include <stdio.h>

int main(){

    return 0;
}

size_t strlen(const char * s){
    size_t n = 0;
    while (*s++){
        n++;
    }
    return n; 
}


char *strcat(char *s1, const char *s2){
    char *p = s1;
    while (*p++);
    while ((*p++ = *s2++));   // assign s2 to p and increment until '\0' is reached, which evaluates to false 
    return s1;
}
