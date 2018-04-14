#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/socket.h>
#include <netinet/in.h>


#ifndef PORT
  #define PORT 30000
#endif
#define MAX_BACKLOG 5
#define MAX_CONNECTIONS 12
#define BUF_SIZE 128


struct sockname {
    int sock_fd;
    char *username;
};


/* Accept a connection. Note that a new file descriptor is created for
 * communication with the client. The initial socket descriptor is used
 * to accept connections, but the new socket is used to communicate.
 * Return the new client's file descriptor or -1 on error.
 */
int accept_connection(int fd, struct sockname *usernames) {
    int user_index = 0;
    // find vacant location in usernames[]
    while (user_index < MAX_CONNECTIONS && usernames[user_index].sock_fd != -1) {
        user_index++;
    }

    int client_fd = accept(fd, NULL, NULL);
    if (client_fd < 0) {
        perror("server: accept");
        close(fd);
        exit(1);
    }

    if (user_index == MAX_CONNECTIONS) {
        fprintf(stderr, "server: max concurrent connections\n");
        close(client_fd);
        return -1;
    }

    // read in name for incoming connection 
    int num_read;
    char *name = malloc(sizeof(char) * (BUF_SIZE + 1));
    if((num_read = read(client_fd, name, BUF_SIZE)) == -1){
        perror("server:read");
        close(fd);
        exit(1);
    }
    name[num_read - 1] = '\0';

    usernames[user_index].sock_fd = client_fd;
    usernames[user_index].username = name;

    printf("Accepted connection from %s\n", name);
    return client_fd;
}


/* Read a message from client_index and echo it back to them.
 * Return the fd if it has been closed or 0 otherwise.
 */
int read_from(int client_index, struct sockname *usernames) {
    int fd = usernames[client_index].sock_fd;
    char buf[BUF_SIZE + 1];

    int num_read = read(fd, &buf, BUF_SIZE);
    buf[num_read] = '\0';

    // append message from client with client username
    char ret_msg[BUF_SIZE + 1] = {'\0'};
    strncpy(ret_msg, usernames[client_index].username, BUF_SIZE);
    strncat(ret_msg, ":", BUF_SIZE - strlen(ret_msg));
    strncat(ret_msg, buf, BUF_SIZE - strlen(ret_msg));

    // Broadcast to all connecting clients 
    int i = 0;
    int curr_fd;
    while(usernames[i].sock_fd != -1 && i < MAX_CONNECTIONS){
        curr_fd = usernames[i].sock_fd;

        int num_written = write(curr_fd, ret_msg, strlen(ret_msg));

        if (num_read == 0 || num_written != strlen(ret_msg)) {
            usernames[i].sock_fd = -1;
            return fd;
        }
        i++;
    }

    return 0;
}


int main(void) {
    int on = 1, status;
    struct sockname usernames[MAX_CONNECTIONS];
    // default value initialization 
    for (int index = 0; index < MAX_CONNECTIONS; index++) {
        usernames[index].sock_fd = -1;
        usernames[index].username = NULL;
    }

    // Create the socket FD.
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("server: socket");
        exit(1);
    }
    
    // Make sure we can reuse the port immediately after the server terminates.
    status = setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR,
              (const char *) &on, sizeof(on));
    if(status == -1) {
        perror("setsockopt -- REUSEADDR");
    }

    // Set information about the port (and IP) we want to be connected to.
    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    server.sin_addr.s_addr = INADDR_ANY;

    // This should always be zero. On some systems, it won't error if you
    // forget, but on others, you'll get mysterious errors. So zero it.
    memset(&server.sin_zero, 0, 8);

    // Bind the selected port to the socket.
    if (bind(sock_fd, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("server: bind");
        close(sock_fd);
        exit(1);
    }

    // Announce willingness to accept connections on this socket.
    if (listen(sock_fd, MAX_BACKLOG) < 0) {
        perror("server: listen");
        close(sock_fd);
        exit(1);
    }

    // The client accept - message accept loop. First, we prepare to listen to multiple
    // file descriptors by initializing a set of file descriptors.
    int max_fd = sock_fd;
    fd_set all_fds, listen_fds;
    FD_ZERO(&all_fds);
    FD_SET(sock_fd, &all_fds);              // add the server fd to the set 

    while (1) {
        // select updates the fd_set it receives, so we always use a copy and retain the original.
        listen_fds = all_fds;
        int nready = select(max_fd + 1, &listen_fds, NULL, NULL, NULL);
        if (nready == -1) {
            perror("server: select");
            exit(1);
        }

        // Is it the original socket? Create a new connection ...
        // You can actually use select for observing if socket has incoming connection
        if (FD_ISSET(sock_fd, &listen_fds)) {
            int client_fd = accept_connection(sock_fd, usernames);
            if (client_fd > max_fd) {
                max_fd = client_fd;
            }
            FD_SET(client_fd, &all_fds);
        }

        // Next, check the clients.
        // NOTE: We could do some tricks with nready to terminate this loop early.
        for (int index = 0; index < MAX_CONNECTIONS; index++) {
            if (usernames[index].sock_fd > -1 && FD_ISSET(usernames[index].sock_fd, &listen_fds)) {
                // Note: never reduces max_fd
                int client_closed = read_from(index, usernames);
                if (client_closed > 0) {
                    FD_CLR(client_closed, &all_fds);
                    printf("Client %d disconnected\n", client_closed);
                } else {
                    printf("Echoing message from client %d\n", usernames[index].sock_fd);
                }
            }
        }
    }


    // Should never get here.
    return 1;
}
