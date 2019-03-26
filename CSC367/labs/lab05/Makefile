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

MPICC = mpicc
CFLAGS += -std=gnu11 -Wall -Werror -g3 -O3 -DNDEBUG

all: transpose-sg transpose-alltoall

transpose-sg: transpose-sg.o
	$(MPICC) $^ -o $@

transpose-alltoall: transpose-alltoall.o
	$(MPICC) $^ -o $@

%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o transpose-sg transpose-alltoall
