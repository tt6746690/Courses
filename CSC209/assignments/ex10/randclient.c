#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

#ifndef PORT
  #define PORT 30000
#endif

#define TIMES 5 // number of times to send the message
#define MINCHARS 3
#define MAXCHARS 7

int main(int argc, char** argv) {
  int soc;
  char message[18] = "A stitch in time\r\n";
  struct sockaddr_in peer;              // address of server in this case

  int current_byte, bytes_left, total_bytes, howmany;
  char piece[MAXCHARS];

  // Create endpoint for current process
  if ((soc = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    perror("randclient: socket");
    exit(1);
  }

  // Create address of peer socket
  peer.sin_family = AF_INET;                                // IPv4 network, address in ddd.ddd.ddd.ddd
  peer.sin_port = htons(PORT);                              // access server port at PORT 
  if (inet_pton(AF_INET, argv[1], &peer.sin_addr) < 1) {    // convert address to struct in_addr
    perror("randclient: inet_pton");
    close(soc);
    exit(1);
  }

  // Initiate 3-way handshake to connect soc to peer. 
  if (connect(soc, (struct sockaddr *)&peer, sizeof(peer)) == -1) {
    perror("randclient: connect");
    exit(1);
  }
  // Connection established; Now we can read and write to soc 

  total_bytes = TIMES * sizeof(message);
  current_byte = 0;
  while (current_byte < total_bytes) {
    // randomly chose a way between MAXCHARS anb MINCHARS
    howmany = rand() % (MAXCHARS - MINCHARS + 1) + MINCHARS;
    bytes_left = total_bytes - current_byte;
    if (howmany > bytes_left) {
      howmany = bytes_left;
    }
    // populate piece with next _howmany_ char from message 
    for (int i = 0; i < howmany; i++) {
      piece[i] = message[(current_byte + i) % sizeof(message)];
    }
    // write to socket created by socket() previously 
    write(soc, piece, howmany);
    current_byte += howmany;
  }
  close(soc);
  return 0;
}
