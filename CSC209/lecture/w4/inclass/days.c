#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(){
  char first[7] = "Monday";
  char *second = "Tuesday"; // string literal
  char *third = malloc(strlen("Wednesday") + 1); // use strlen instead

  /*

  *third = "Wednesday";   // assigning position of string literal to pointer
                          // same as *second = "Tuesday"
                          // Is WRONG

  */

  /* first arg  - container to copy into
                - make sure there is enough memory
     second arg - string to copy (source)
     third arg  - CAPACITY of first arg
                - how much memory allocated for first arg
  */
  strncp(third, "Wednesday", 10);
  third[9] = '\0';      // terminates at last index

  // strcp() // buffer overflow bug if use strcp instead

  // change Monday to Mon
  first[3] = '\0';    // add null terminator for truncation
  second[3] = '\0';    // doesnt work because string in read only memory


  return 0;
}
