# This illustrates how you can set the value of a defined constant 
# on the compile line.
PORT=30001
CFLAGS = -DPORT=$(PORT) -g -Wall -std=gnu99

all: inetserver inetclient 

# The variable $@ has the value of the target. 
inetserver: inetserver.o
	gcc ${CFLAGS} -o $@ inetserver.o  

inetclient: inetclient.o
	gcc ${CFLAGS} -o $@ inetclient.o 

forkserver: forkserver.o
	gcc ${CFLAGS} -o $@ forkserver.o 
simpleselect: simpleselect.o 
	gcc ${CFLAGS} -o $@ simpleselect.o 

.c.o:
	gcc  ${CFLAGS}  -c $<

clean:
	rm *.o 

realclean:
	rm inetserver inetclient forkserver 
