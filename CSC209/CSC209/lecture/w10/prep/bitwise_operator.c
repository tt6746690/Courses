#include <stdio.h>

int main() {
    // gcc allows you to enter binary constants by prefacing the
    // the number with 0b:
    char a = 0b00010011;    // decimal value: 19

    // Similarly, preface with 0x for hexadecimal constants:
    unsigned char b = 0x14; // decimal value: 20

    // Negation:
    printf("result of negative %x is %x in hex\n", a, ~a);

    // Bitwise AND:
    printf("result of bitwise AND of %x and %x is %x in hex\n",
        a, b, a & b);

    // Bitwise OR:
    printf("result of bitwise OR of %x and %x is %x in hex\n",
        a, b, a | b);

    // Bitwise XOR:
    printf("result of bitwise XOR of %x and %x is %x in hex\n",
        a, b, a ^ b);

    return 0;
}