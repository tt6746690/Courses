/***** inetserver.c *****/ 
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdlib.h>        /* for getenv */
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>    /* Internet domain header */

#ifndef PORT
#define PORT 30000
#endif

int main()
{ 
    int soc, ns, k;
    int on = 1, status;
    char buf[256];
    struct sockaddr_in peer;
    struct sockaddr_in self; 
    int peer_len = sizeof(peer);
    char *host;

    self.sin_family = AF_INET;
    self.sin_port = htons(PORT);  
    printf("Listening on %d\n", PORT);
    self.sin_addr.s_addr = INADDR_ANY;
    bzero(&(self.sin_zero), 8);

    printf("PORT=%d\n", PORT);

    peer.sin_family = AF_INET;
    /* set up listening socket soc */
    soc = socket(AF_INET, SOCK_STREAM, 0);
    if (soc < 0) {  
	perror("server:socket"); 
	exit(1);
    }

    status = setsockopt(soc, SOL_SOCKET, SO_REUSEADDR,
        (const char *) &on, sizeof(on));
    if(status == -1) {
	perror("setsockopt -- REUSEADDR");
    }
    if (bind(soc, (struct sockaddr *)&self, sizeof(self)) == -1) {  
	 perror("server:bind"); close(soc);
	 exit(1); 
    }
    listen(soc, 1);                              
    /* accept connection request */
    printf("Calling accept\n");
    ns = accept(soc, (struct sockaddr *)&peer, &peer_len);          
    if (ns < 0) {  
	perror("server:accept"); 
	close(soc);
	exit(1);
    }
    /* data transfer on connected socket ns */
    k = read(ns, buf, sizeof(buf));
    host = getenv("HOST");
    printf("SERVER ON %s RECEIVED: %s\n", host, buf);
    write(ns, buf, k);
    close(ns);   
    close(soc);
    return(0);
}
