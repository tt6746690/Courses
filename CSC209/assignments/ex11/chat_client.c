#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
 

#ifndef PORT
  #define PORT 30000
#endif
#define BUF_SIZE 128

int main(void) {
    // Create the socket FD.
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("client: socket");
        exit(1);
    }

    // Set the IP and port of the server to connect to.
    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    if (inet_pton(AF_INET, "127.0.0.1", &server.sin_addr) < 1) {
        perror("client: inet_pton");
        close(sock_fd);
        exit(1);
    }



    // Connect to the server.
    if (connect(sock_fd, (struct sockaddr *)&server, sizeof(server)) == -1) {
        perror("client: connect");
        close(sock_fd);
        exit(1);
    }

    // pass in username
    char username[BUF_SIZE];
    char message[] = "enter your username: ";
    write(STDOUT_FILENO, message, sizeof(message));
    int n_read = read(STDIN_FILENO, username, BUF_SIZE);
    if(write(sock_fd, username, n_read) != n_read){
        perror("client:write");
        close(sock_fd);
        exit(1);
    }

    // Read input from the user, send it to the server, and then accept the
    // echo that returns. Exit when stdin is closed.
    char buf[BUF_SIZE + 1];
    int num_select;
    int num_read;
    int num_written;

    // Use select to multiplex STDIN and socket
    // Without select, one of two fd is always blocking.. 
    // have to enter (i.e. read from stdin) to be able to see 
    // what other client has sent, which lead to other client 
    // reasing empty messages 
    int max_fd = (STDIN_FILENO > sock_fd) ? STDIN_FILENO : sock_fd;
    fd_set all_fd, read_fds;

    FD_ZERO(&all_fd);
    FD_SET(STDIN_FILENO, &all_fd);
    FD_SET(sock_fd, &all_fd);

    while (1) {
        // resets mod_fd;
        read_fds = all_fd;

        num_select = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (num_select == -1) {
            perror("client: select");
            exit(1);
        }

        if(FD_ISSET(STDIN_FILENO, &read_fds)){
            num_read = read(STDIN_FILENO, buf, BUF_SIZE);
            if (num_read == 0) {
                break;
            }
            buf[num_read] = '\0';         // Just because I'm paranoid

            num_written = write(sock_fd, buf, num_read);
            if (num_written != num_read) {
                perror("client: write");
                close(sock_fd);
                exit(1);
            }
        }

        if(FD_ISSET(sock_fd, &read_fds)){
            num_read = read(sock_fd, buf, BUF_SIZE);
            buf[num_read] = '\0';
            printf("Received from server: \n%s", buf);
        }

    }

    close(sock_fd);
    return 0;
}
