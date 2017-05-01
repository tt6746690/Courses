#include <stdio.h>
#include <limits.h>

int main() {

    // Create an int and a double and just print them.
    int i = 17;
    double d = 4.8;
    printf("i is %d\n", i);	
    printf("d is %f\n", d);

    // What happens when we assign a double to an int?
    i = d;
    printf("i is %d\n", i);	// 4; double after decimal is truncated to give an int 
    // Mostly it just drops the fractional piece,
    // unless the double value is outside the range of the integer.
    printf("but if double is bigger than the largest int\n");
    d = 3e10;
    printf("d is %f\n", d);
    i = d;			//  -2147483648 ; d is larger than i can hold here  
    printf("i is %d\n", i);

    // What about assigning an integral type to a floating point type?
    i = 17;
    d = i;			// 17.000000   still a float; because int is a subset of float  
    printf("d is %f\n", d);

    // What if it is a large unsigned int
    // and we assign it to a floating point type of the same size (in bytes)?
    
    printf("An integer is stored using %lu bytes \n", sizeof(i));	// 4 bytes 
    printf("A double is stored using %lu bytes \n", sizeof(d));		// 8 bytes 
    float f;
    printf("A float is stored using %lu bytes \n", sizeof(f));		// 4 bytes 

    int big = INT_MAX; 					// INT_MAX is the largest int value possible 
    printf("big is %d uses %lu bytes \n", big, sizeof(big));  // big = 2147483647 
    f = big;
    printf("f is %f uses %lu bytes \n", f, sizeof(f)); 	// f =  2147483648.000000
   							// note its off by 1... 

    // Conversion between sizes of integral types.
    char ch = 'A';
    printf("char ch: displayed as char %c, displayed as int %d\n", ch, ch);

    int j = ch;
    printf("j is %c, int %d\n", j, j);		// j is A, int 65 
	
    i = 320;				// here i is larger than char can store 
    ch = i;
    printf("char ch: displayed as char %c, displayed as int %d\n", ch, ch);	// char ch: displayed as char @, displayed as int 64  = 320 - 256, which is @ in ASCII 
   
    return 0;
}


   
