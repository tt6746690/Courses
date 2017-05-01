#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int
dowork(int id, int in)
{
  char buf[256];
  int n;
  int num;
  while ((n = read(in, buf, 256)) > 0 ){
    num = atoi(buf);
    printf("%d: %d\n", id, num / 2);
  }
  exit(num / 2);
}


int
main(int argc, char **argv)
{
  int i, j, status, n;
  int fd[5][2];
  char buf[256];

  for (i = 0; i < 5; i ++) {
    pipe(fd[i]);
    if (fork() == 0) {
      for (j = 0; j < i; j++) {
        close(fd[j][0]);
        close(fd[j][1]);
      }
      close(fd[i][1]);
      dowork(i, fd[i][0]);
    }
  }

  for (i = 0; i < 5; i++) {
    n = sprintf(buf, "%d", i*4);
    write(fd[i][1], buf, n);
  }

  /* don't forget to close the pipes first! (this code way below) */

  for (i = 0; i < 5; i++) {
    int pid;
    if ((pid = wait(&status)) == -1) {
      perror("wait");
    } else {
      if (WIFEXITED(status)) {
        printf("Proces %d exited with %d\n", pid, WEXITSTATUS(status));
      }
    }
  }
  return 0;
}






















/*
  for (i = 0; i < 5; i++) {
    close(fd[i][1]);
  }
*/
