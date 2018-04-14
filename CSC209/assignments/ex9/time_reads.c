#include <stdio.h> 
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

#include <sys/time.h>

int iteration;
int s;

void handler(int code){
    fprintf(stderr, "Read speed = [%d] per [%d] second", iteration, s);
    exit(0);
}


int main(int argc, char **argv){

    if(argc != 3){
        fprintf(stdout, "Usage test_reads time filename");
        exit(1);
    }

    s = strtol(argv[1], NULL, 10);

    // adding sigaction 
    struct sigaction sa;
    sa.sa_handler = handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGALRM, &sa, NULL);


    // set timer 
    struct itimerval new_value;
    struct timeval alert_time;      // struct for representing time 
    alert_time.tv_sec = s;          // seconds
    alert_time.tv_usec = 0;         // microseconds
    new_value.it_value = alert_time;        // start decrementing from it_value
    new_value.it_interval = alert_time;     // reset to it_interval after it_value reach 0 and signal generated
    if(setitimer(ITIMER_REAL, &new_value, NULL) == -1){
        perror("setitimer");
        exit(1);
    }

    // mask SIGALRM for subsequent procedures
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);

    // Reads an integer randomly from 0 ~ 100 in an infinity loop
    FILE *f;
    if((f = fopen(argv[2], "r+")) == NULL){
        perror("fopen");
        exit(1);
    }

    int number;
    int r;
    for(;;iteration++){
        sigprocmask(SIG_BLOCK, &set, NULL);

        r = random() % 99; 
        if(fseek(f, r * sizeof(int), SEEK_SET) == -1){
            perror("fseek");
            exit(1);
        }
        if(fread(&number, sizeof(int), 1, f) != 1){
            fprintf(stdout, "Error reading from file");
            exit(1);
        }
        /* if(fwrite(&r, sizeof(int), 1, f) != 1){ */
        /*     fprintf(stdout, "Error writing to file"); */
        /*     exit(1); */
        /* } */

        /* printf("%d ", number); */
        sigprocmask(SIG_UNBLOCK, &set, NULL);
    }

    exit(0);
}
