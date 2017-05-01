
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char **parity_strings(const char *s){
    int even_size = 0;
    int odd_size = 0;

    // Note that sizesof(s) should not be used since they are size in memory 
    // should use strlen instead 

    for (int i =0; i< strlen(s); i++){
        if(i%2 == 0){
            even_size ++;
        } else {
            odd_size ++;
        }
    }
                
    char **two_string = malloc(sizeof(char *)*2);
    two_string[0] = malloc(sizeof(char) * (even_size+ 1));
    two_string[1] = malloc(sizeof(char) * (odd_size+1));

    int j = 0;
    int k = 0;
    for (int i = 0; i < strlen(s); i++){
        if(i%2 == 0){
            two_string[0][j] = s[i];
            j++;
        } else {
            two_string[1][k] = s[i];
            k++;
        }
    }

    two_string[0][even_size] = '\0';
    two_string[1][odd_size] = '\0';
    return two_string;
}


int main(int argc, char **argv){
    char **r = parity_strings(argv[1]);
    printf("%s, %s, %s", r[0], r[1], argv[1]);
    return 0;
}
   
