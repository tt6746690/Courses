#include <stdio.h>
#include "ftree.h"


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage:\n\tfcopy SRC DEST\n");
        return 0;
    }

    int ret = copy_ftree(argv[1], argv[2]);
    if (ret < 0) {
        printf("Errors encountered during copy\n");
        ret = -ret;
    } else {
        printf("Copy completed successfully\n");
    }
    printf("%d processes used\n", ret);
    
    return 0;
}

