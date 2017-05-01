#include <stdio.h>

int main() {
    char text[20] = {'h', 'e', 'l', 'l', 'o', '\0'};
    // This does the same thing
    //char text[20] = "hello";
  
    printf("%s\n", text);


    // But text1 is a pointer to a string literal
    char *text1 = "hello";
    printf("%s\n", text1);


    // We can change a character in text but not text1    
    text[0] = 'j';
    printf("%s\n", text);
    // See what happens if we uncomment the next line and compile
    // text1[3] = 'X';

    return 0;
}
