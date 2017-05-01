#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

char *NAME;

void sing(int code){
    fprintf(stdout, "Happy Birthday %s", NAME);
    return;
}

int main(int argc, char *argv[]){

    if (argc != 2){
        fprintf(stdout, "Usage: greeting NAME");
    }

    // memory leak 
    /* NAME = malloc(strlen(argv[1]) + 1); */
    /* strcpy(NAME, argv[1]); */
    NAME = argv[1];

    struct sigaction act;
    act.sa_handler = sing;
    act.sa_flags = 0;
    sigemptyset(&act.sa_mask);
    // sa_mask holds the set of signals blocked 
    sigaction(SIGQUIT, &act, NULL);

    for(;;){
        fprintf(stdout, ".");
        usleep(100000);
    }
    return 0;
}
