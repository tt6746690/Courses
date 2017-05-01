#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int i;
    struct stat sbuf;
    char fullname[NAME_MAX];

    if(argc < 2) {
	fprintf(stderr, "Usage: printdirs [dir ...]\n");
	exit(1);
    }

    for(i = 1; i < argc; i++) {
	DIR *dp = opendir(argv[i]);
	struct dirent *entry;

	if (dp == NULL) {
	    perror(argv[i]);
	    continue;
	}
	strncpy(fullname, argv[i], NAME_MAX);
	while((entry = readdir(dp)) != NULL) {
	    strncpy(fullname, argv[i], NAME_MAX);
	    strcat(fullname, "/");
	    strcat(fullname, entry->d_name);

	    if(stat(fullname, &sbuf) == -1) {
		perror(entry->d_name);
		continue;
	    }
	    if(S_ISDIR(sbuf.st_mode)) {
		printf("%s\n", fullname);
	    }
	}
	closedir(dp);
    }
    return 0;
}
