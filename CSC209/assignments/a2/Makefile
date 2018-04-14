FLAGS = -Wall -std=gnu99
HEADERS = hash.h ftree.h

all: print_ftree

# concat all compiled file to one  
print_ftree: print_ftree.o ftree.o hash_functions.o
	gcc ${FLAGS} -o $@ $^

# compiles .c to .o file if header files present
%.o: %.c ${HEADERS}
	gcc ${FLAGS} -o $@ -c $<

clean: 
	rm *.o print_ftree

.PHONY: clean 
