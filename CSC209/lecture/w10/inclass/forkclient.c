/*
** client.c -- a stream socket client demo
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>

#ifndef PORT
#define PORT 11491 /* the port client will be connecting to  */
#endif

#define MAXDATASIZE 100 /* max number of bytes we can get at once  */
void Writen(int, void *, size_t);
ssize_t Readn(int, void *, size_t);
ssize_t Readline(int, void *, size_t);
int main(int argc, char *argv[])
{
    int sockfd, numbytes;  
    char buf[MAXDATASIZE];
    struct hostent *he;
    struct sockaddr_in their_addr; /* connector's address information  */

    if (argc != 2) {
        fprintf(stderr,"usage: client hostname\n");
        exit(1);
    }

    if ((he=gethostbyname(argv[1])) == NULL) {  /* get the host info  */
        perror("gethostbyname");
        exit(1);
    }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(1);
    }

    their_addr.sin_family = AF_INET;    /* host byte order */
    their_addr.sin_port = htons(PORT);  /* short, network byte order */
    their_addr.sin_addr = *((struct in_addr *)he->h_addr);
    bzero(&(their_addr.sin_zero), 8);   /* zero the rest of the struct */

    if (connect(sockfd, (struct sockaddr *)&their_addr, 
		sizeof(struct sockaddr)) == -1) {
        perror("connect");
        exit(1);
    }

    printf("Reading from server\n");
    numbytes = Readline(sockfd, buf, MAXDATASIZE);
    printf("Returned from read %d\n", numbytes);

    buf[numbytes] = '\0';

    printf("%s\n", buf);
    fgets(buf, MAXDATASIZE, stdin);
    Writen(sockfd, buf, strlen(buf) + 1);

    close(sockfd);

    return 0;
}