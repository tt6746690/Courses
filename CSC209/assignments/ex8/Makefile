all : validate checkpasswd

validate : validate.c
	gcc -Wall -g -o validate validate.c

checkpasswd : checkpasswd.c
	gcc -Wall -g -o checkpasswd checkpasswd.c

clean:
	rm validate checkpasswd

