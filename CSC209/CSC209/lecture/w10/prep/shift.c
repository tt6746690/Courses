#include <stdio.h>

// Return var with the kth bit set to 1
unsigned char setbit(unsigned char var, int k) {
	return var | (1 << k);
}

// Return 0 if the kth bit in var is 0,
// and a non-zero value if the kth bit is 1
int checkbit(unsigned char var, int k) {
	return var & (1 << k);
}

int main() {
    unsigned char b = 0xC1; //1100 0001 in binary

	printf("Original value %x\n", b);

	// set the third bit of b to one
	b = b | 0x8; // 0x8 == 0000 1000
	printf("With the third bit set %x\n", b);

	// check if the second bit of b is set
	// 1100 1001
	if(b & 0x4) { // if the result of bitwise AND-ing b with 0x8 is not zero
		printf("The second bit of %x is 1\n", b);
	} else {
		printf("The second bit of %x is 0\n", b);
	}
	
    return 0;
}