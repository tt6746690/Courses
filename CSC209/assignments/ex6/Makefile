listfiles : listfiles.c
	gcc -std=c99 -Wall -g -o listfiles listfiles.c

test : listfiles 
	./listfiles testdir > testdir.actual
	./compare_result testdir
