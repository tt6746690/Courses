#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char **argv) {
    struct stat sbuf;

    if (stat(argv[1], &sbuf) == -1) {
	    perror("stat");
        exit(1);
    } else {
	    printf("Size of %s is %lld\n", argv[1], sbuf.st_size);
    }

    return 0;
}