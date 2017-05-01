#include <stdio.h>

/* Changes the value of the 0th element of int array numbers
 * to 80. 
 */
void change(int numbers[]) {
    numbers[0] = 80;
}

int main() {
    int my_array[5];

    my_array[0] = 40;
    change(my_array);

    // The following prints 80 because "change" modifies the
    // *original* array, since the array was *not* copied
    // when passed as an argument:
    printf("Element at index 0: %d\n", my_array[0]);
  
    return 0;
}






















