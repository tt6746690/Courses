#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>

#ifndef PORT
  #define PORT 30000
#endif

int setup(void) {
  int on = 1, status;
  struct sockaddr_in self;
  int listenfd;
  if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    perror("socket");
    exit(1);
  }

  // Make sure we can reuse the port immediately after the
  // server terminates.
  status = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
                      (const char *) &on, sizeof(on));
  if(status == -1) {
    perror("setsockopt -- REUSEADDR");
  }

  self.sin_family = AF_INET;
  self.sin_addr.s_addr = INADDR_ANY;
  self.sin_port = htons(PORT);
  memset(&self.sin_zero, 0, sizeof(self.sin_zero));  // Initialize sin_zero to 0

  printf("Listening on %d\n", PORT);

  if (bind(listenfd, (struct sockaddr *)&self, sizeof(self)) == -1) {
    perror("bind"); // probably means port is in use
    exit(1);
  }

  if (listen(listenfd, 5) == -1) {
    perror("listen");
    exit(1);
  }
  return listenfd;
}

/*Search the first inbuf characters of buf for a network newline ("\r\n").
  Return the location of the '\r' if the network newline is found,
  or -1 otherwise.
  Definitely do not use strchr or any other string function in here. (Why not?)
    > buf is a byte stream, which may contain null character before 
    the first inbuf chars, hence using string function will terminate before 
   entirety inbuf 
*/

int find_network_newline(const char *buf, int inbuf) {
  // Step 1: write this function
    char chr;
    
    for(int i = 0; i < inbuf; i++){
        chr = *(buf + i);
        if(chr == '\r'){
            return i;
        }
    }

  return -1; // return the location of '\r' if found
}

int main() {
  int listenfd;
  int fd, nbytes;
  char buf[30];
  int inbuf; // how many bytes currently in buffer?
  int room; // how much room left in buffer?
  char *after; // pointer to position after the (valid) data in buf
  int where; // location of network newline

  struct sockaddr_in peer;
  socklen_t socklen;

  listenfd = setup();
  while (1) {
    socklen = sizeof(peer);
    // Note that we're passing in valid pointers for the second and third
    // arguments to accept here, so we can actually store and use client
    // information.
    if ((fd = accept(listenfd, (struct sockaddr *)&peer, &socklen)) < 0) {
      perror("accept");

    } else {
      printf("New connection on port %d\n", ntohs(peer.sin_port));

      // Receive messages
      inbuf = 0;          // buffer is empty; has no bytes
      room = sizeof(buf); // room == capacity of the whole buffer
      after = buf;        // start writing at beginning of buf

      while ((nbytes = read(fd, after, room)) > 0) {
        // Step 2: update inbuf (how many bytes were just added?)
        inbuf += nbytes;
        
        // Step 3: call find_network_newline, store result in variable "where"
        where = find_network_newline(buf, inbuf);
        
        printf("inbuf = [%d], room = [%d]\n", inbuf, room);
        if (where >= 0) { // OK. we have a full line

          // Step 4: output the full line, not including the "\r\n",
          // using print statement below.
          // Be sure to put a '\0' in the correct place first;
          // otherwise you'll get junk in the output.
          // (Replace the "\r\n" with appropriate characters so the 
          // message prints correctly to stdout.)
          *(buf + where) = '\0';
          
          printf("Next message: %s\n", buf);
          // Note that we could have also used write to avoid having to
          // put the '\0' in the buffer. Try using write later!
          
          
          // Step 5: update inbuf and remove the full line from the buffer
          // There might be stuff after the line, so don't just do inbuf = 0
          inbuf = inbuf - where - 2;
          memset(&buf, '\0', where + 2);        // clear buf before and inlcuding \r\n
          
          // You want to move the stuff after the full line to the beginning 
          // of the buffer.  A loop can do it, or you can use memmove.
          // memmove(destination, source, number_of_bytes)
          memmove(buf, buf + where + 2, 30 - where - 2);

        }
        // Step 6: update room and after, in preparation for the next read
        room = 30 - inbuf;
        after = buf + inbuf; 

        printf("inbuf = [%d], room = [%d]\n", inbuf, room);

       
      }
      close(fd);
    }
  }
  return 0;
}
   
