FLAGS = -Wall -g -std=c99
all : simplefork forkloop parentcreates childcreates

simplefork : simplefork.o
	gcc ${FLAGS} -o simplefork simplefork.o

forkloop : forkloop.o
	gcc ${FLAGS} -o $@ $^

parentcreates : parentcreates.o
	gcc ${FLAGS} -o $@ $^

childcreates : childcreates.o
	gcc ${FLAGS} -o $@ $^

simplepipe : simplepipe.o
	gcc ${FLAGS} -o simplepipe simplepipe.o

%.o : %.c
	gcc ${FLAGS} -c $<

clean :
	rm *.o simplefork forkloop parentcreates childcreates
