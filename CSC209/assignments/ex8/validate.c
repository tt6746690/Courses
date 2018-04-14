#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#define MAXLINE 256

#define PASSWORD_FILE "pass.txt"

/* Reads two chunks from stdin, and checks if they match a user id
and password pair from a password file. The first chunk (10 bytes)
will contain a user id, and the second chunk (10 bytes) will contain a password.
The program exits with a value of 0 if the user id and password match, 
1 if there is an error, 2 if the user id is found but the password does not match, and 3 if the user id is not found in the password file. */

int main(void){
    int n, user_length;
    char userid[30];
    char password[10];
    
    if ((n = read(STDIN_FILENO, userid, 10)) == -1) {
        perror("read");
        exit(1);
    } else if(n == 0) {
        fprintf(stderr, "Error: could not read from stdin");
        exit(1);
    } 
    if (userid[n-1] == '\n')
        userid[n-1] = '\0';
    else
        userid[n] = '\0';

    if ((n = read(STDIN_FILENO, password, 10)) == -1) {
        perror("read");
        exit(1);
    } else if (n == 0) {
        fprintf(stderr, "Error: could not read from stdin");
        exit(1);
    } 
    if (password[n-1] == '\n')
        password[n-1] = '\0';
    else
        password[n] = '\0';

    strcat(userid, ":");
    user_length = strlen(userid);
    strcat(userid, password);
    FILE *fp = fopen(PASSWORD_FILE, "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }
    char line[MAXLINE];
    while(fgets(line, sizeof(line) - 1, fp)) {
        line[strlen(line) - 1] = '\0';
        if (strcmp(userid, line) == 0)
            exit(0); // found match
        else if(strncmp(userid, line, user_length) == 0)
            exit (2); // invalid password
    }
    exit(3); // no such user
}
