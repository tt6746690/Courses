#include <stdio.h> 
#include <stdlib.h>

#define N 10

int read_line(char *str, int size);
int count_spaces(char *str);

int main(){

    char *str = malloc(N);
    read_line(str, N);
    printf("input string is %s", str);
    printf(" number of spaces = %d", count_spaces(str));
    return 0;

}


int read_line(char *str, int size){
    int ch, i = 0;

    while ( (ch = getchar()) != '\n'){
        if (i < size){
            str[i++] = ch;
        }
    }
    str[i] = '\0';
    return i;       // number of strings stored 
}


int count_spaces(char *str){
    int count = 0;
    
    for (; *str != '\0'; str++){
        if (*str == ' '){
            count ++;
        }
    }
    return count;
}
