PORT=52562
CFLAGS = -DPORT=$(PORT) -g -Wall -std=gnu99

all: chat_server chat_client

chat_server: chat_server.o
	gcc ${CFLAGS} -o $@ $^

chat_client: chat_client.o
	gcc ${CFLAGS} -o $@ $^

%.o: %.c
	gcc ${CFLAGS} -c $<

clean:
	rm -f *.o chat_server chat_client
