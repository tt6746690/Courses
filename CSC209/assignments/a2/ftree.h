#ifndef _FTREE_H_
#define _FTREE_H_

/*
 * Data structure for storing information about a single file.
 * For directories, contents is the linked list of files in the directory and hash is NULL.
 * For files, contents is NULL, and the hash is the hash of the file's contents.
 * next is the next file in the directory (or NULL).
 */
struct TreeNode {
    char *fname;
    int permissions;

    struct TreeNode *contents;   // For directories
    char *hash;                  // For normal files and links

    struct TreeNode *next;
};


/*
 * A FTree is a dynamically allocated tree structure that contains
 * information about the files in a file system. A FTree is represented by
 * a single TreeNode which is the root of the tree.
 */

// Function for generating a FTree given a root filename.
struct TreeNode *generate_ftree(const char *fname);

// Function for printing the TreeNodes encountered on a preorder traversal of a FTree.
void print_ftree(struct TreeNode *root);

#endif // _FTREE_H_
