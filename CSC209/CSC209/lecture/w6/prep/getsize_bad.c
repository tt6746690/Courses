#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char **argv) {
    struct stat sbuf;

    stat(argv[1], &sbuf);
    printf("Size of %s is %lld\n", argv[1], sbuf.st_size);

    return 0;
}