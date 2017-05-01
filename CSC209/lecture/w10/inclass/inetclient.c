/***** inetclient.c *****/ 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>    
#include <netdb.h>

#ifndef PORT
#define PORT 30000
#endif

int main(int argc, char* argv[])
{ 
    int soc;
    char buf[256];
    struct hostent *hp;
    struct sockaddr_in peer;

    peer.sin_family = PF_INET;
    peer.sin_port = htons(PORT); 
    printf("PORT = %d\n", PORT);


    if ( argc != 2 )
    {  
        fprintf(stderr, "Usage: %s hostname\n", argv[0]);
        exit(1);
    }

    /* fill in peer address */
    hp = gethostbyname(argv[1]);                
    if ( hp == NULL ) {  
        fprintf(stderr, "%s: %s unknown host\n",
            argv[0], argv[1]);
        exit(1);
    }

    peer.sin_addr = *((struct in_addr *)hp->h_addr);

    /* create socket */
    if((soc = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
		perror("socket");
	}
    /* request connection to server */
    if (connect(soc, (struct sockaddr *)&peer, sizeof(peer)) == -1) {  
        perror("client:connect"); close(soc);
        exit(1); 
    }
    if(write(soc, "Hello Internet\r\n", 17) == -1) {            // soc is socket descriptor..
                                                                // \r\n used to transmit string over connection 
		perror("write");
		exit(1);
	}
    if(read(soc, buf, sizeof(buf)) <= 0) {
		perror("read");
		exit(1);
	}
    printf("SERVER ECHOED: %s\n", buf);
    if(close(soc) == -1) {
		perror("close");
	}
    return(0);
}
