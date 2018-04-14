# We will eventually discuss Makefiles in class, and you will learn how to
# write your own Makefile.
# You can use this file to build your program without having to type so much.
# IF you type "make" the appropriate gcc commands will be run to produce the
# compute_hash executable program.

# You do not need to change or submit this file

FLAGS = -Wall -std=c99 -g

compute_hash: compute_hash.o hash_functions.o
	gcc ${FLAGS} -o $@ $^

%.o : %.c
	gcc ${FLAGS} -c $<

clean :
	rm *.o compute_hash
