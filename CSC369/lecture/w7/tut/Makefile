CCFLAGS = -Wall -g

all : readimage

readimage : readimage.o utilities.o
	gcc $(CCFLAGS) -o readimage $^

%.o : %.c ext2.h utilities.h
	gcc $(CCFLAGS) -c $<


clean : 
	rm -f *.o 
	rm -f readimage
