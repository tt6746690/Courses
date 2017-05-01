#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
8 && 7
#define INTSIZE 32
#define N 4

struct bits {
    unsigned int field[N];
};

typedef struct bits Bitarray;

/* Initialize the set b to empty or all zeros
 * Return 0 if memset fails.
 */
int setzero(Bitarray *b){
  return (memset(b, 0, sizeof(Bitarray)) == NULL);
}

/* Add value to the set b
 *   (set the bit at index value in b to one)
 */
void set(unsigned int value, Bitarray *b) {
    int index = value / INTSIZE;
    b->field[index] |= 1 << (value % INTSIZE);
}

/* Remove value from the set b
 *   (set the bit at index value in b to zero)
 */
void unset(unsigned int value, Bitarray *b) {
    int index = value / INTSIZE;
    b->field[index] &= ~(1 << (value % INTSIZE));
}

/* Return true if value is in the set b, and false otherwise.
 *    Return a non-zero value if the bit at index 'value' is one in b
 *    Return zero if the bit at index 'value' is zero in b
 */
int ifset(unsigned int value, Bitarray *b) {
    int index = value / INTSIZE;
    return ( (1 << (value % INTSIZE)) & b->field[index]);
}

/* Run some simple tests on the above functions*/
int main() {

    Bitarray a1;
    setzero(&a1);

    // Add 1, 16, 32, 65 to the set
    set(1, &a1);
    set(16, &a1);
    set(32, &a1);
    set(68, &a1);

    // Expecting: [ 0x00010002, 0x00000001, 0x000000010, 0 ]
    // Print using hexadecimal
    printf("%x %x %x %x\n",
            a1.field[0], a1.field[1], a1.field[2], a1.field[3]);

    unset(68, &a1);

    // Expecting: [ 0x00010002, 0x00000001, 0, 0 ]
    // Print using hexadecimal 
    printf("%x %x %x %x\n",
            a1.field[0], a1.field[1], a1.field[2], a1.field[3]);

    return 0;
}
