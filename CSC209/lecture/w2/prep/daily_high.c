#include <stdio.h>

int main() {

    // we declare an array of 4 float values
    float daytime_high[4];

    // we initialize these values one at a time
    // later we will learn other ways to do this
    daytime_high[0] = 16.0;
    daytime_high[1] = 12.8;
    daytime_high[2] = 14.6;
    daytime_high[3] = 19.1;
    
    
    float average_temp = (daytime_high[0] + daytime_high[1] + daytime_high[2] + daytime_high[3]) / 4;

    // the variable index is used to access one of the array elements
    int index = 1;
    printf("On day %d, the high was %f.\n",index, daytime_high[index]);

    return 0;
}