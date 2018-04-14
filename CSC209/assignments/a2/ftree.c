#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ftree.h"
#include "hash.h"

#define PATH_MAX 4096

/*
 * Returns the FTree rooted at the path fname.
 */
struct TreeNode *generate_ftree(const char *fname) {

    struct TreeNode *node = malloc(sizeof(struct TreeNode));
    struct stat *stat_buf = malloc(sizeof(struct stat));
    // Static path variable keeping track of relative path from root 
    static char path[PATH_MAX];
    strncat(path, fname, sizeof(path) - strlen(path) - 1);

    /*
     * Populate members of node with given / default values 
     */
    // Assign fname
    int name_length = strlen(fname);
    node->fname = malloc(name_length + 1);
    strncpy(node->fname, fname, name_length);
    node->fname[name_length] = '\0';

    // Fetch stat buffer file at path
    if(lstat(path, stat_buf) != 0){        // on failure
        perror("lstat");
        return NULL;
    }

    // Assign permissions
    node->permissions = stat_buf->st_mode & 0777;
    // Assign default NULL to contents, next, and hash
    node->contents = NULL;
    node->next = NULL;
    node->hash = NULL;

    /*
     * If fname is a directory, loop through its contents and recursively build ftree
     *  which would be stored in a linked list under node->contents
     * Otherwise fname is a regular file or a link, compute hash function from file's content
     */
    if (S_ISDIR(stat_buf->st_mode)){                
        DIR *dirp = opendir(path);
        struct dirent *dp;

        if(dirp == NULL) {
            perror("opendir");
            return NULL;
        }
        // appends forward slash for directory type file 
        strncat(path, "/", sizeof(path) - strlen(path) - 1); 

        while ((dp = readdir(dirp)) != NULL){             // access file in current directory
            if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0){     // avoid . and .. files
                /* printf("path = [%s] ... fname = [%s] ... file type = [%d]\n", path, dp->d_name, dp->d_type); */
                struct TreeNode *new_node = generate_ftree(dp->d_name);
                if (new_node != NULL){                    // skips current node if returns a NULL pointer
                    if (node->contents == NULL){          // check for first insert
                        node->contents = new_node;
                    } else {                              // otherwise loop through linked list and append new node to the end
                        struct TreeNode *cur = node->contents;
                        for(; cur->next != NULL; cur = cur->next){}
                        cur->next = new_node;
                    }
                }
            }
        }
        // resets path static variable to parent directory  
        path[strlen(path) - strlen(fname) - strlen("/")] = '\0';
    } else {  
        FILE *f;
        if ((f = fopen(path, "r")) == NULL){
            perror("fopen");
            return NULL;
        }
        // Assigns hash value of current file to node 
        node->hash = hash(f);
        if (fclose(f) != 0) {
            perror("fclose");
            return NULL;
        }
        // resets path static variable to parent directory
        path[strlen(path) - strlen(fname)] = '\0';
    } 
    free(stat_buf);
    return node;
}


void print_ftree(struct TreeNode *root) {
    // Here's a trick for remembering what depth (in the tree) you're at
    // and printing 2 * that many spaces at the beginning of the line.
    static int depth = 0;
    printf("%*s", depth * 2, "");
    
    if(root == NULL){
        exit(EXIT_FAILURE);
    }

    if(root->hash == NULL){     // is directory
        printf("===== %s (%o) =====\n", root->fname, root->permissions);
        struct TreeNode *cur = root->contents;
        if(cur != NULL){
            depth++;                    // increment depth if root is a directory
            for(; cur->next != NULL; cur = cur->next){
                print_ftree(cur);
            }
            print_ftree(cur);           // the last node with node->next == NULL is still valid to print
            depth--;                     
        }
    } else {
        printf("%s (%o)\n", root->fname, root->permissions);
    }

}
