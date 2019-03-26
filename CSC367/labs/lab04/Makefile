# ------------
# This code is provided solely for the personal and private use of
# students taking the CSC367 course at the University of Toronto.
# Copying for purposes other than this use is expressly prohibited.
# All forms of distribution of this code, whether as given or with
# any changes, are expressly prohibited.
#
# Authors: Bogdan Simion, Maryam Dehnavi, Alexey Khrabrov
#
# All of the files in this directory and all subdirectories are:
# Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
# -------------

CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -g3 -fopenmp -O3 -DNDEBUG
LDFLAGS += -fopenmp -lm

all: primes-seq primes-default primes-static primes-dynamic integral-seq integral-omp

primes-seq: primes-seq.o
	$(CC) $^ -o $@ $(LDFLAGS)

primes-default: primes-default.o
	$(CC) $^ -o $@ $(LDFLAGS)

primes-static: primes-static.o
	$(CC) $^ -o $@ $(LDFLAGS)

primes-dynamic: primes-dynamic.o
	$(CC) $^ -o $@ $(LDFLAGS)

integral-seq: integral-seq.o
	$(CC) $^ -o $@ $(LDFLAGS)

integral-omp: integral-omp.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o primes-seq primes-default primes-static primes-dynamic integral-seq integral-omp
