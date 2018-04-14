/*********
 * CLIENT
 *********/

#ifndef _CLIENT_H_
#define _CLIENT_H_


#include <unistd.h>     // close read write
#include <string.h>     // memset
#include <errno.h>      // perror
#include <stdlib.h>     // exit
#include <sys/stat.h>   // stat
#include <dirent.h>     // readdir DIR
#include <netdb.h>      // sockaddr_in
#include <sys/wait.h>   // wait


#include "hash.h"       // hash()
#include "ftree.h"      // request stuct



/* Create a new socket that connects to host
 * Waiting for a successful connection
 * Returns sock_fd and exits should error arises
 */
int client_sock(char *host, unsigned short port);

/* Construct client request for file/dir at path
 * request is modified to accomodate changes
 * Return 0 on success and -1 on failure
 */
int make_req(const char *path, const char *server_path, struct request *client_req);

/*
 * Sends request struct to sock_fd over 5 read calls
 * In order of
 * -- type
 * -- path
 * -- mode
 * -- hash
 * -- size
 * Return 0 if success -1 otherwise
 */
int send_req(int sock_fd, struct request *req);

/*
 * precondition: req.st_mode yields regular file
 * Sends data specified by req by
 * -- open file at req.path
 * -- write to client socket where nbytes is
 * ---- BUFSIZE if eof is not reached
 * ---- position of EOF if eof is reached
 * Return 0 if success -1 otherwise
 */
int send_data(int fd, const char *client_path, struct request *req);
//void send_data(int fd, struct request *req);

/*
 * Recursively traverses filepath rooted at source with sock_fd
 * Then for each file or directory
 * -- makes and sends request struct
 * -- waits for response from server
 * ---- OK: continue to next file
 * ---- SENDFILE:
 * ------ forks new process, main process continues to next file, child:
 * ------ initiate new connection with server w/e request.type = TRANSFILE
 * ------ makes and sends request struct
 * ------ transmit data from file
 * ------ waits for OK, close socket and exits, otherwise handles error
 * ----ERROR: print appropriate msg includes file name then exit(1)
 * Return
 * -- -1 for any error
 * -- 0 for success
 * -- >0 the number of child processes created
 */
//int traverse(const char *source, int sock_fd, char *host, unsigned short port);
int traverse(const char *source, const char *server_dest, int sock_fd, char *host, unsigned short port);

/*
 * The main client waits for count number of
 * child processes to terminate 
 * Return 0 if success -1 otherwise
 */
int client_wait();

#endif
