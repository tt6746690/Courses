#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: write_test_file filename\n");
        exit(1);
    }
    FILE * fp;
    if ((fp = fopen(argv[1],"w")) == NULL) {
        perror("fopen");
        exit(1);
    }
    int i, j;

    for (i=0; i<1000; i++) {
        j = random() % 100;
        if (fwrite(&j, 1, sizeof(int), fp) != sizeof(int)) {
            perror("write");
            exit(1);
        }
    }

    fclose(fp);
    return 0;
}
