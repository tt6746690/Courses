/*********
 * SERVER
 *********/

#ifndef _SERVER_H_
#define _SERVER_H_

#include <unistd.h>     // close read write
#include <string.h>     // memset
#include <errno.h>      // perror
#include <stdlib.h>     // exit
#include <sys/stat.h>   // stat
#include <netdb.h>      // sockaddr_in

#include "hash.h"       // hash()
#include "ftree.h"      // request stuct


/*
 * linked list for tracking mult read for sending request struct
 *
 * -- fd
 * ---- client's file descriptor
 * -- current_state
 * ---- AWAITING_TYPE 0
 * ---- AWAITING_PATH 1
 * ---- AWAITING_SIZE 2
 * ---- AWAITING_PERM 3
 * ---- AWAITING_HASH 4
 * ---- AWAITING_DATA 5
 * -- file
 * ---- keeps stream open for copying files over multiple read calls
 * -- request
 * ---- request sent from client
 * -- next
 * ---- pointer to the next client in the linked list
 */
struct client {
    int fd;
    int current_state;
    FILE *file;
    struct request client_req;
    struct client *next;
};

/*
 * Allocates memory for a new struct client
 * at end of linked list with given fd
 * Returns 
 * -- pointer to the newly created element if success
 * -- NULL pointer otherwise
 */
struct client * linkedlist_insert(struct client *head, int fd);

/*
 * Delete client in head linked list with given fd
 * Return element before deleted item if found; NULL otherwise
 */
struct client *linkedlist_delete(struct client *head, int fd);

/*
 * Print linked list at head
 * Each node is presented as fd
 */
void linkedlist_print(struct client *head);

/*
 * Creates server socket
 * binds to PORT and starts litening to
 * connection from INADDR_ANY
 *
 * program terminate upon sys call failures
 */
int server_sock(unsigned short port);

/*
 * Reads request struct from client to cli over 5 write calls
 * In order of
 * -- type
 * -- path
 * -- mode
 * -- hash
 * -- size
 * Returns
 * -- fd if
 * ---- file transfer socket finish transfer file
 * ---- main socket finish traversing filepath
 * -- 0 to continue reading req
 * -- -1 if sys call fails
 */
int read_req(struct client *cli);

/*
 * Compare files based on client request (cli->client_req)
 * Sends res signal to client
 * SENDFILE
 * -- server_file does not exist
 * -- server_file different in hash from client_file
 * OK
 * -- server_file and client_file are identical
 * ERROR
 * -- file types are incompatible (i.e. file vs. directory)
 * Return
 * -- ERROR on sys call error 
 * -- SENDFILE if file / dir does not match request
 * -- OK if file / dir match request
 */
int compare_file(struct client *cli);

/*
 * Makes directory given client request with given
 * -- path
 * -- permission
 * Return -1 on error and fd if success
 */
int make_dir(struct client *cli);

/*
 * Makes file given client request with given
 * -- path
 * -- permission
 * Return
 * -- -1 on error
 * -- 0 if file copy not finished
 * -- fd if file copy finished
 * (i.e. file transfer over multiple select calls)
 */
int make_file(struct client *cli);

/*
 * Writes inputs from client socket to destination file
 * on the server.
 * Precondition: client has a valid input file stream
 *      created using make_file
 * Postcondition: input file stream is closed only when
 *      client has finished sending the file over
 * Return
 * -- -1 on error
 * -- 0 if file copy not finished
 * -- fd if file copy finished
 * (i.e. file transfer over multiple select calls)
 */
int write_file(struct client *cli);

#endif // _SERVER_H_
