#include <stdio.h>
#include "ftree.h"


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage:\n\tftree DIRECTORY\n");
        return 0;
    }

    struct TreeNode *root = generate_ftree(argv[1]);
    print_ftree(root);

    return 0;
}

