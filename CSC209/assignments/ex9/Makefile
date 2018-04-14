FLAGS = -Wall -std=gnu99 -g
DEPENDENCIES = hash.h ftree.h

all: time_reads

test: make_test_file.c 
	gcc ${FLAGS} -o $@ $<

time_reads: time_reads.c 
	gcc ${FLAGS} -o $@ $<


clean: 
	rm test time_reads

