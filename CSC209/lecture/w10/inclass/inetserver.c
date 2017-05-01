/***** inetserver.c *****/ 
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>        /* for getenv */
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>    /* Internet domain header */

#ifndef PORT                // for giving PORT from gcc -DPORT option
#define PORT 30000
#endif

int main()
{ 
    int listenfd, newsoc, k;
    int on = 1, status;
    char buf[256];
    struct sockaddr_in peer;
    struct sockaddr_in self; 
    unsigned int peer_len = sizeof(peer);
    char host[256];

    self.sin_family = PF_INET;  // allow sockets across machines
    self.sin_port = htons(PORT); // which port will we be listening on 
    printf("Listening on %d\n", PORT);
    self.sin_addr.s_addr = INADDR_ANY; // listen on all network addresses
    bzero(&(self.sin_zero), 8);  

    printf("PORT=%d\n", PORT);

    peer.sin_family = PF_INET;
    /* set up listening socket soc */
    listenfd = socket(PF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) {  
        perror("server:socket"); 
        exit(1);
    }

    // Make sure we can reuse the port immediately after the
    // server terminates. Avoids the "address in use" error
    status = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
        (const char *) &on, sizeof(on));
    if(status == -1) {
        perror("setsockopt -- REUSEADDR");          // OS will keep closed port in table for a small period 
                                                    // such that OS can provide error for access. 
                                                    // this is to enable using the same port right after closing it.
    }

    // Associate the process with the address and a port
    if (bind(listenfd, (struct sockaddr *)&self, sizeof(self)) == -1) {  
        perror("server:bind"); close(listenfd);
        exit(1); 
    }

    // Sets up a queue in the kernel to hold pending connections
    listen(listenfd, 1);                              

    /* accept connection request */
    printf("Calling accept\n");
    newsoc = accept(listenfd, (struct sockaddr *)&peer, &peer_len);          
    if (newsoc < 0) {  
        perror("server:accept"); 
        close(listenfd);
        exit(1);
    }

    /* data transfer on connected socket ns */
    k = read(newsoc, buf, sizeof(buf));
    if(k == -1) {
        perror("socket read");
    } else if (k == 0) {
        fprintf(stderr, "Client closed\n");
    }
    if((gethostname(host, 256)) == -1) {
        perror("gethostname");
    }

    printf("SERVER ON %s RECEIVED: %s\n", host, buf);
    if((write(newsoc, buf, k)) < k) {
        fprintf(stderr, "Error: could not write all the data\n");
    };
    if(close(newsoc) == -1) {
        perror("close");   
    }
    if(close(listenfd)) {
        perror("close");
    }
    return(0);
}
