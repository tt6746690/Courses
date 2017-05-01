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

    peer.sin_family = AF_INET;
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
    soc = socket(AF_INET, SOCK_STREAM, 0);
    /* request connection to server */
    if (connect(soc, (struct sockaddr *)&peer, sizeof(peer)) == -1)
    {  
	perror("client:connect"); close(soc);
	exit(1); 
    }
    write(soc, "Hello Internet\n", 16);           
    read(soc, buf, sizeof(buf));
    printf("SERVER ECHOED: %s\n", buf);
    close(soc); 
    return(0);
}
