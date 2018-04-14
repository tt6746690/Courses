PORT=52561
CFLAGS = -DPORT=$(PORT) -g -Wall -std=gnu99

all: readserver randclient bufserver

readserver: readserver.o
	gcc ${CFLAGS} -o $@ readserver.o

randclient: randclient.o
	gcc ${CFLAGS} -o $@ randclient.o

bufserver: bufserver.o
	gcc ${CFLAGS} -o $@ bufserver.o

%.o: %.c
	gcc ${CFLAGS} -c $<

clean:
	rm *.o readserver randclient bufserver
