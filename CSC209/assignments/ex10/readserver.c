#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>

#ifndef PORT
  #define PORT 30000
#endif

int setup (void) {
  int on = 1;
  struct sockaddr_in self;                                      // current address
  int listenfd;                                                 // socket descriptor for server
  if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    perror("socket");
    exit(1);
  }

  // Make sure we can reuse the port immediately after the server terminates.
  if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
                 &on, sizeof(on)) == -1) {
    perror("setsockopt -- REUSEADDR");
  }

  memset(&self, '\0', sizeof(self));                            // initializes `self` to NULL
  self.sin_family = AF_INET;                                    // Creates socket address for listenfd
  self.sin_addr.s_addr = INADDR_ANY;                            // binds socket to all local interfaces 
  self.sin_port = htons(PORT);                                  // Converts Port to network byte order 
  printf("Listening on %d\n", PORT);

  if (bind(listenfd, (struct sockaddr *)&self, sizeof(self)) == -1) {   // Assigns addr to listenfd socket 
    perror("bind"); // probably means port is in use
    exit(1);
  }

  if (listen(listenfd, 5) == -1) {                              // Starts accepting incoming connection
    perror("listen");
    exit(1);
  }
  return listenfd;
}

int main(void) {
  int listenfd;
  int fd, nbytes;
  struct sockaddr_in peer;
  socklen_t socklen;
  char buf[20];

  listenfd = setup();
  while (1) {
    socklen = sizeof(peer);
    // Note that we're passing in valid pointers for the second and third
    // arguments to accept here, so we can actually store and use client
    // information.
    if ((fd = accept(listenfd, (struct sockaddr *)&peer, &socklen)) < 0) {  // Extract connection from peer 
                                                                            // fd is socket descriptor for peer socket 
      perror("accept");
    } else {
      printf("New connection on port %d\n", ntohs(peer.sin_port));
      
      // Receive messages
      while ((nbytes = read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[nbytes] = '\0';
        printf ("Next message: [%s]\n\n", buf);
      }
      close(fd);
    }
  }
}
   
