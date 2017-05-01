#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

/* A signal handling function that simply prints
   a message to standard error. */
void handler(int code) {
    fprintf(stderr, "Signal %d caught\n", code);
}

int main() {
    // Declare a struct to be used by the sigaction function:
    struct sigaction newact;

    // Specify that we want the handler function to handle the
    // signal:
    newact.sa_handler = handler;

    // Use default flags:
    newact.sa_flags = 0;

    // Specify that we don't want any signals to be blocked during
    // execution of handler:
    sigemptyset(&newact.sa_mask);

    // Modify the signal table so that handler is called when
    // signal SIGINT is received:
    sigaction(SIGINT, &newact, NULL);

    // Keep the program executing long enough for users to send
    // a signal:
    int i = 0;
    
    for (;;) {
        if ((i++ % 50000000) == 0) {
            fprintf(stderr, ".");
        }
    }

    return 0;
}