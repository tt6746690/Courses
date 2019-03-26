CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -g3 -fopenmp -O3 -DNDEBUG
LDFLAGS += -fopenmp -lm

all: test

primes-seq: test.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o test
