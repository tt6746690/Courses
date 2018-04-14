#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <libgen.h>

#include "ftree.h"
#include "hash.h"

#define PATH_MAX 4096
#define BUFFER_SIZE 258

void compare_and_copy(const char *src, const char *dest);

/*
   Copies file and directory at file tree rooted at src 
   to file tree rooted at dest. Creates and copies file 
   from src to dest if file is not existent. Otherwise 
   updates data and permissions for dest files from src
   */
int copy_ftree(const char *src, const char *dest){
    struct stat buf;                    // stat buf for src file
    int process_count = 1;              // tallies total number of processes 

    char src_path[PATH_MAX] = {0};             
    char dest_path[PATH_MAX] = {0};
    char *src_copy = strdup(src);               // needs to free 
    char *src_basename = basename(src_copy);    // arg requires char * not const char *

    // appends src basename to dest
    strncpy(src_path, src, sizeof(src_path) - strlen(src_path) - 1);
    strncpy(dest_path, dest, sizeof(dest_path) - strlen(dest_path) - 1);
    strncat(dest_path, "/", sizeof(dest_path) - strlen(dest_path) - 1);
    strncat(dest_path, src_basename, sizeof(dest_path) - strlen(dest_path) - 1);
    free(src_copy);
    /* printf("src_path = [%s] dest_path = [%s]\n", src_path, dest_path); */

    // Fetch stat buffer for src file
    if(lstat(src_path, &buf) != 0){        
        perror("lstat");
        return(process_count);
    }   

    /*
       Evaluate current file specified by src_path
       -- If file is a directory, recursively call copy_ftree, fork child process for child directories
       -- If file is a reguar file, copy src to dest if their size or hash does not match
       */
    if(S_ISDIR(buf.st_mode)){
        DIR *dirp = opendir(src_path);
        struct dirent *dp;

        if(dirp == NULL) {
            perror("opendir");
            return(process_count);
        }

        /*
         * Check if dest file exists, create directory at dest 
         * if no such file exists. Otherwise, check file type, 
         * report missmatch and stop copy if dest file is a regular file
         * or update directory with given permission if dest file is a directory
       */
        struct stat dest_buf;
        int src_perm = buf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);

        DIR *dest_dir = opendir(dest_path);

        if(lstat(dest_path, &dest_buf) == -1){
            // dest file/directory does not exist
            if(errno == ENOENT){
                /* printf("<<< create directory at dest [%s] with permission [%o] >>> \n", dest_path, src_perm); */
                if(mkdir(dest_path, src_perm) == -1) {
                    perror("mkdir");
                    return(process_count);
                }
            } else {
                perror("lstat");
                return(process_count);
            }
        } else {
            // dest file/directory exists
            if(S_ISREG(dest_buf.st_mode)){
                fprintf(stderr, "Missmatch error: dest and src has different file type but same file name, skip\n");
                return(process_count);
            } else if(S_ISDIR(dest_buf.st_mode)){
                /* printf("<<< update directory at dest [%s] with permission [%o] >>> \n", dest_path, src_perm); */
                if(chmod(dest_path, src_perm) == -1){
                    fprintf(stderr,  "cannot set file [%s] permission", dest_path);
                }
                if(closedir(dest_dir) == -1){
                    perror("closedir");
                    return(process_count);
                }
            }
        }


        strncat(src_path, "/", sizeof(src_path) - strlen(src_path) - 1); 

        /* 
         * Recursively call copy_ftree on all files in this directory 
         * -- if dp is a child directory, fork a child and then call copy_ftree
         * -- if dp is a regular file, call copy_ftree
         */
        int child_count = 0;                                // keep track of number of subdirectories
        while ((dp = readdir(dirp)) != NULL){           
            if (dp->d_name[0] != '.'){                      // avoid files starting with .
                // building src path for files inside src
                strncat(src_path, dp->d_name, sizeof(src_path) - strlen(src_path) - 1);
                if(dp->d_type == DT_DIR){                   // dp is a directory
                    child_count++;
                    pid_t result = fork();

                    if(result < 0){
                        perror("fork");
                        exit(EXIT_FAILURE);
                    } else if(result == 0){                 // child process
                        /* printf("child process [%d] created for directory = [%s]\n", getpid(), src_path); */
                        int num = copy_ftree(src_path, dest_path);
                        exit(num);                          // stops recursively spawning child processes
                    }

                } else {                                    // dp is a regular file 
                    copy_ftree(src_path, dest_path);
                }
                // resetting to directory path 
                src_path[strlen(src_path) - strlen(dp->d_name)] = '\0';
            }
        }
        // resetting to src_path to src
        src_path[strlen(src_path) - strlen("/")] = '\0';

        if(closedir(dirp) == -1) {
            perror("closedir");
            return(process_count);
        }

        // parent wait for children and collects process count 
        while(child_count-- != 0){
            pid_t pid;
            int status;
            if((pid = wait(&status)) == -1) {
                perror("wait");
                exit(EXIT_FAILURE);
            } else {
                if (WIFEXITED(status)) {
                    /* printf("Child %d terminated with %d [PROCESS_COUNT = (%d)]\n", pid, WEXITSTATUS(status), process_count); */
                    process_count += WEXITSTATUS(status);
                }
            }
        }

    } else if(S_ISREG(buf.st_mode)){         
        compare_and_copy(src_path, dest_path);
    } 

    return(process_count);
}


/*
   compare size and hash of src and dest file, 
   copy src to dest if dest does not exist 
   or if either the size or hash of both files do not match 
   */
void compare_and_copy(const char *src, const char *dest){
    int bytes;
    FILE *src_file, *dest_file;
    char buffer[BUFFER_SIZE];

    struct stat src_buf, dest_buf;
    char *src_hash, *dest_hash;

    if((src_file = fopen(src, "r+")) == NULL){           // src file always exist
        perror("fopen src");
        return;
    } 

    /* retrieve stat buffer for src and dest file */
    if(lstat(src, &src_buf) != 0){        
        perror("lstat");
        return;
    }

    /*
       If dest file does not exists, copy src to dest and function returns
       If dest file exists, check for dest file type 
       if dest is a directory, report missmatch and stop copy, 
       if dest is a regular file, open dest file for reading 
       */

    if(lstat(dest, &dest_buf) != 0){                   
        if(errno == ENOENT){
            if((dest_file = fopen(dest, "w")) == NULL){   
                perror("fopen");
                return;
            }
            if(chmod(dest, src_buf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO)) == -1){
                fprintf(stderr,  "cannot set file [%s] permission\n", dest);
            }
            /* printf("=== copy src [%s] to dest [%s] (create new file) === \n", src, dest); */
            while((bytes = fread(buffer, 1, BUFFER_SIZE, src_file)) > 0){
                fwrite(buffer, 1, bytes, dest_file);
            }
            // free fildes 
            if(fclose(src_file) != 0){
                perror("fclose");
                return;
            }
            if(fclose(dest_file) != 0){
                perror("fclose");
                return;
            }
            return;
        } else {
            perror("lstat");
            return;
        }
    } else {

        if(S_ISDIR(dest_buf.st_mode)){
            fprintf(stderr, "Missmatch error: dest and src has different file type but same file name, skip\n");
            return;
        } 
        if((dest_file = fopen(dest, "r+")) == NULL){   
            // skip current file if not accessible
            if (errno == EACCES){                                      
                fprintf(stderr, "dest file permission denied\n");
                return;
            }
            perror("fopen");
            return;
        } 
        if(chmod(dest, src_buf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO)) == -1){
            fprintf(stderr,  "cannot set file [%s] permission", dest);
        }
    }

    /*
       If src and dest file size does not match, copy src to dest and function returns
       otherwise check file hash, copy src to dest if hash does not match.
       */
    if(src_buf.st_size != dest_buf.st_size){              
        // reopen dest from r+ to w+ for overwriting pre-existing dest content
        /* printf("=== copy src [%s](%lld) to dest [%s](%lld) (size different)===\n", src, (long long)src_buf.st_size, dest, (long long)dest_buf.st_size); */
        freopen(dest, "w+", dest_file);                  
        while((bytes = fread(buffer, 1, BUFFER_SIZE, src_file)) > 0){
            fwrite(buffer, 1, bytes, dest_file);
        }
    } else {                                            
        src_hash = hash(src_file);
        dest_hash = hash(dest_file);  
        rewind(src_file);           // reset file pointer to beginning of file

        if(strcmp(src_hash, dest_hash) != 0){           
            // reopen dest from r+ to w+ for overwriting pre-existing dest content
            /* printf("=== copy src [%s](%s) to dest [%s](%s)  (hash different)===\n", src, src_hash, dest, dest_hash); */
            freopen(dest, "w+", dest_file);                 
            while((bytes = fread(buffer, 1, BUFFER_SIZE, src_file)) > 0){
                fwrite(buffer, 1, bytes, dest_file);
            }
        }                                              
    }


    // free fildes 
    if(fclose(src_file) != 0){
        perror("fclose");
        return;
    }
    if(fclose(dest_file) != 0){
        perror("fclose");
        return;
    }
    return;
}







