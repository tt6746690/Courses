
# define MAX_NAME_SIZE 32
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


struct player {
    char name[MAX_NAME_SIZE];
    char *position;
    int home_runs;
    float avg;
};


void out_of_the_park(struct player *p){
}

void f(struct player p){
    p.position[0]= 'D';
}


int main(){

    // declare struct p1 
    struct player p1;
    strncpy(p1.name, "Josh Donaldson", MAX_NAME_SIZE);
    p1.name[MAX_NAME_SIZE-1] = '\0';    // very last char is null termination 

    p1.position = malloc(strlen("third base")+1);
    // if do not want to mutate position: position holds address on read-only memory  
    // p.position = "third baase";
    strncpy(p1.position, "third base", strlen("third base"));
    p1.position[strlen("third base")] = '\0';

    return 0;
}
