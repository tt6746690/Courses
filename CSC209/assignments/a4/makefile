PORT=52945
CFLAGS = -DPORT=$(PORT) -g -Wall -std=gnu99
HEADERS = hash.h ftree.h client.h server.h


all: rcopy_client rcopy_server

rcopy_client: rcopy_client.o ftree.o hash_functions.o client.o server.o
	gcc ${CFLAGS} -o $@ $^

rcopy_server: rcopy_server.o ftree.o hash_functions.o client.o server.o
	gcc ${CFLAGS} -o $@ $^


%.o: %.c ${HEADERS}
	gcc ${CFLAGS} -c $<

clean:
	rm *.o 

server:
	chmod 777 sandbox && rm -r sandbox
	clear
	./rcopy_server .

client:
	clear
	./rcopy_client adir localhost

debug:
	chmod 777 sandbox && rm -r sandbox 
	gdb ./rcopy_server
