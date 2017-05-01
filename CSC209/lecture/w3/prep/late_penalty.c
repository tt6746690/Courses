#include <stdio.h>

/* This function is supposed to lower the grade by one letter-grade. 
 * It doesn't work correctly. 
 */
void apply_late_penalty(char grade) {
   if (grade != 'F') {
       grade++;
   }
}

/* This version works correctly */
void properly_apply_late_penalty(char *grade_pt) {
   if (*grade_pt != 'F') {
       (*grade_pt)++;
   }
}

int main() {
   
    char grade_Michelle = 'B';
    printf("Michelle earned a %c\n", grade_Michelle);

    // We can add 1 to the char 'B' and get the next char in sequence.
    // Since the ASCII codes come in alphabetical order, this would be a 'C'
    grade_Michelle++;   
    printf("Michelle was late, so instead she gets a %c\n",grade_Michelle);

    // Felipe was also late on his assignment but he started with an A
    char grade_Felipe = 'A';
    printf("Felipe earned a %c\n", grade_Felipe);

    //  Let's call a function to lower Felipe's mark.
    apply_late_penalty(grade_Felipe);

    // Felipe's grade didn't change
    printf("Felipe was also late and earns a %c\n",grade_Felipe);


    // Let's call our improved function to lower Felipe's mark
    properly_apply_late_penalty(&grade_Felipe);			// note address is passed as argument to become the value of local param pointer grade_pt:
    printf("Felipe was also late and earns a %c\n",grade_Felipe);

    return 0;
}
