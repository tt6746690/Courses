/* 
 * utils.c - definitions of utility functions for CSC 209 A3 Summer 2006
 *
 * If you wish to use this file, be sure to add it to your CVS repository
 * and add Makefile rules to link utils.o to your program.
 *
 * You may modify this file for this assignment however you wish.
 */


#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include "utils.h"

#define NUMPERM 10000

/*Returns a string that corresponds to an int between 1 and 52
  representing a card
  Warning: This function does not allocate memory for the string*/

char *sprintcard(int num, char *string){
  switch(num % 13){
  case 1:
    string[0] = 'A';
    break;
  case 10:
    string[0] = 'T';
    break;
  case 11:
    string[0] = 'J';
    break;
  case 12:
    string[0] = 'Q';
    break;
  case 0:
    string[0] = 'K';
    break;
  default:
    string[0] = num % 13 + '0';
    break;
  }
  switch((num - 1) / 13){
  case 0:
    string[1] = 'C';
    break;
  case 1:
    string[1] = 'D';
    break;
  case 2:
    string[1] = 'H';
    break;
  case 3:
    string[1] = 'S';
    break;
  }
  string[2] = '\0';
  return string;
}

/*Returns a char that corresponds to a string of the form 6H (6 of Hearts)*/

char cardtoc(char *card){
  char value = 0;

  switch(card[0]){
  case 'A':
    value += 1;
    break;
  case 'T':
    value += 10;
    break;
  case 'J':
    value += 11;
    break;
  case 'Q':
    value += 12;
    break;
  case 'K':
    value += 13;
    break;
  default:
    if(card[0] < '1'|| card[0] > '9')
      return -1;
    value += card[0] - '0';
  }

  switch(card[1]){
  case 'C':
    break;
  case 'D':
    value += 13;
    break;
  case 'H':
    value += 26;
    break;
  case 'S':
    value += 39;
    break;
  }

  return value;
}

/*Returns the value of a hand of num cards
  Returns -1 on error.*/

int valuecards(char *cards, int num){
  int total = 0, i, numaces = 0;
  
  for (i = 0; i < num; i++)
    switch (cards[i] % 13) {
    case 10:
    case 11:
    case 12:
    case 0:
      total += 10;
      break;
    case 1:
      total += 11;
      numaces++;
      break;
    default:
      total += cards[i] % 13;
    }

  while ((total > 21) && (numaces > 0)){
    total -= 10;
    numaces--;
  }

  return total;
}

/*Seeds the pseudo-random number generator, and return the seed that was used.
  Returns 0 on failure.
  Note: 0 can also be returned in the unlikely case (1/(2^32)) where it is 
  the value read from /dev/urandom. We just consider that 0 is not a good
  seed.
*/

unsigned int seedprng(unsigned int seed){
  int fd;

  if (!seed){
    fd = open("/dev/urandom", O_RDONLY);
    
    if(read(fd, &seed, 4) != 4){
      close(fd);
      return 0;
    }

    close(fd);
  }

  srandom(seed);

  return (unsigned int) seed;
}

/*Returns a random number between 0 and range - 1.*/

static int getrand(int range){
  int num;

  num = (int) random();

  return (int) ( ((double) num / RAND_MAX) * range );
}

/*Given a pointer to an array of 52*numdecks chars, and numdecks,
  fills the array with cards, in a (somewhat) random way.*/

void shuffledeck(char *deck, int numdecks){
  int i, j;
  
  for (i = 0; i < numdecks; i++)
    for (j = 0; j < 52; j++)
      deck[52*i + j] = j + 1;

  for (i = 0; i < NUMPERM; i++){
    int a, b;
    char temp;

    a = getrand(52*numdecks);
    b = getrand(52*numdecks);
    temp = deck[a];
    deck[a] = deck[b];
    deck[b] = temp;
  }
}

/*Outputs the cards from position begin to position end, comma separated*/

void printcards(char *cards, int begin, int end){
  int i;
  char string[3];

  for (i = begin; i < end; i++)
    fprintf(stderr, "%s,", sprintcard(cards[i], string));
  fprintf(stderr, "%s", sprintcard(cards[i], string));
  fprintf(stderr, "\n");
}
