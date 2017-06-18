all : heaploop matmul

heaploop : heaploop.c
	gcc -Wall -g -o heaploop heaploop.c
matmul : matmul.c
	gcc -Wall -g -o matmul matmul.c

traces: heaploop matmul
	./runit heaploop
	./runit matmul 32

clean : 
	rm heaploop matmul tr-matmul.ref tr-heaploop.ref marker tmp
