#include <stdio.h>
#define DAYS 4

int main() {

    // this does the same thing as the first five lines of
    // daily_highs.c
    float daytime_high[DAYS] = {16.0, 12.8, 14.6, 19.1};
    
    // we initially use this variable to hold the running total
    float average_temp = 0;

    // We loop here over the array and add each element to the running total
    int i;
    for (i = 0; i < DAYS; i++) {
        printf("adding element %d with value %f\n", i, daytime_high[i]);
        average_temp += daytime_high[i];
    }

    // We divide by the number of items. The running total becomes the average.
    average_temp = average_temp / DAYS;
    printf("average %f\n", average_temp);
    return 0;
}









