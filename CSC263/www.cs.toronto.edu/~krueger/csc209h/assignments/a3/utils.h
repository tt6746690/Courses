/* 
 * utils.h - declarations of utility functions for CSC 209 A3 Summer 2006
 *
 * If you wish to use this file, be sure to add it to your CVS repository
 * and add Makefile rules to link utils.o to your program.
 *
 * You may modify this file for this assignment however you wish.
 */

/*Returns a string that corresponds to an int between 1 and 52
  representing a card
  Warning: This function does not allocate memory for the string*/

char *sprintcard(int num, char *string);

/*Returns a char that corresponds to a string of the form 6H (6 of Hearts)*/

char cardtoc(char *card);

/*Returns the value of a hand of num cards
  Returns -1 on error.*/

int valuecards(char *cards, int num);

/*Seeds the pseudo-random number generator, and returns the seed that was used.
  If the seed argument is 0, then a seed is read from /dev/urandom.
  Returns 0 on failure.
  Note: 0 can also be returned in the unlikely case (1/(2^32)) where it is 
  the value read from /dev/urandom. We just consider that 0 is not a good
  seed.
*/

unsigned int seedprng(unsigned int seed);

/*Given a pointer to an array of 52*numdecks chars, and numdecks,
  fills the array with cards, in a (somewhat) random way.*/

void shuffledeck(char *deck, int numdecks);

/*Outputs the cards from position begin to position end, comma separated*/

void printcards(char *cards, int begin, int end);

