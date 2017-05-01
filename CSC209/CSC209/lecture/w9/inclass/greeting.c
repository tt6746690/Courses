#include <stdio.h>
#include <unistd.h>

int main(){
    for(;;){
        fprintf(stderr, ".");
        usleep(100000);
    }
    return 0;
}
