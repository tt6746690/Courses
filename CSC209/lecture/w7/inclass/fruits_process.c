#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main(){

    int i;
    printf("Mangoes\n");
    int r = fork();
    printf("Apples\n");

    if (r == 0) {
        printf("Oranges\n");
        int k = fork();
        if (k >= 0){
            printf("Bananas\n");
        }
    } else if (r > 0){
        printf("Peaches\n");
        for(i = 0; i < 3; i++){
            if((r = fork()) == 0){
                printf("Pears\n");
                exit(0);
                printf("Nectarines\n");
            } else if (r > 0){
                printf("Plumes\n");
            }
        }
    }

            


}
