#include <stdio.h>
#include <unistd.h>

int main() {
    printf("About to call execl. My PID is %d\n", getpid());
    execl("./hello", NULL);
    perror("exec");
    
    return 1;
}