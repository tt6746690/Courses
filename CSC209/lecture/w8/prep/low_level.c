#include <stdio.h>
#include <stdlib.h>

int main() {

    FILE *outfp = fopen("tmpfile", "w");

    if(outfp == NULL) {
        perror("fopen");
        exit(1);
    }

    fprintf(outfp, "This is ");
    fprintf(outfp, "one of several ");
    fprintf(outfp, "calls to fprintf.\n");
    fprintf(outfp, "How many write ");
    fprintf(outfp, "system calls are generated?\n");
    fclose(outfp);
    return 0;
}