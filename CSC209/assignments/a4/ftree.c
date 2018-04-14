#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

#include "server.h"
#include "client.h"

#include "ftree.h"


/*
 * Takes the file tree rooted at source, and copies transfers it to host
 * Return 0 for success and 1 otherwise
 */
int rcopy_client(char *source, char *host, unsigned short port){

    // main socket for tree traversal
    int sock_fd;
    sock_fd = client_sock(host, port);
    if (sock_fd == -1){
        return 1;
    }

    // tree traversal
    int traversed;
    int waited;
    char *base = basename(source);      

    traversed = traverse(source, base, sock_fd, host, port);
    if(traversed == -1){
        fprintf(stderr, "client: traversal failed\n");
    }

    close(sock_fd);
    waited = client_wait();

    if(traversed == -1 || waited == -1){
        return 1;
    }
    return 0;
}


/*
 * Server handles incoming connection
 */
void rcopy_server(unsigned short port){

    int sock_fd;
    sock_fd = server_sock(port);
    printf("Server Starts listening on %d...\n", port);


    // initialize empty fd set for accept
    int max_fd = sock_fd;
    fd_set all_fds, listen_fds;

    FD_ZERO(&all_fds);
    FD_SET(sock_fd, &all_fds);

    // head holds a linked list of client struct
    struct client *head = malloc(sizeof(struct client));

    /*
     * An infinity loop where errors are reported 
     * and the cycle goes to next iteration with continue
     */
    while (1) {
        /* select updates the fd_set it receives,
         * so we always use a copy and retain the original.
         */
        listen_fds = all_fds;

        int nready = select(max_fd + 1, &listen_fds, NULL, NULL, NULL);
        if (nready == -1) {
            perror("server: select");
            continue;
        }

        /* On active server socket, accept incoming client connection
         * Every new client occupies one node in linked list
         */
        if (FD_ISSET(sock_fd, &listen_fds)) {

            int client_fd;
            if ((client_fd = accept(sock_fd, NULL, NULL)) == -1) {
                perror("server: accept");
                continue;
            }

            // update all_fds set
            max_fd = (client_fd > max_fd) ? client_fd : max_fd;
            FD_SET(client_fd, &all_fds);

            // keep track of new client in head
            if (linkedlist_insert(head, client_fd) == NULL){
                continue; 
            }

            printf("Client %d connected\n", client_fd);
        }


        /* Send proper response on active clients in linked list head
         * Note pointer p starts from head->next as the first valid client
         */
        for(struct client*p = head->next; p != NULL; p = p->next){
            if(FD_ISSET(p->fd, &listen_fds)){

                int result = read_req(p);
                /*
                 * result is
                 * -- fd
                 * ---- remove fd from all_fds
                 * ---- close fd
                 * ---- remove client struct from linked list head
                 * -- 0 to continue reading req
                 * -- -1 if sys call fails
                 * ---- report error properly
                 */
                if(result == -1){
                    fprintf(stderr, "server: error on handling file = [%s]\n", (p->client_req).path);
                } else if(result == p->fd){

                    FD_CLR(p->fd, &all_fds);
                    if (close(p->fd) == -1){
                        perror("close socket");
                    }

                    printf("Client %d closed\n", p->fd);

                    /* re-assign pointer p since deletion
                     * invalidates pointers including
                     * and beyond the deleted element */
                    if((p = linkedlist_delete(head, p->fd)) == NULL){
                        fprintf(stderr, "server:linkedlist_delete");
                    }


                }

            }
        }


    }
}
