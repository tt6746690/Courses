#include <stdio.h>
#include <math.h>

long calculate_total(int quantity, double tax_rate);

int main() {
    int order1_owed = calculate_total(125, 0.10);  //expecting 2338 
    int order2_owed = calculate_total(1000, 0.13); // expecting 14690
    int order3_owed = calculate_total(10, 0.15);   // expecting 230

    printf("The amount owed for the first order of ping-pong balls is: %d cents\n", order1_owed);
    printf("The amount owed for the second order of ping-pong balls is: %d cents\n", order2_owed);
    printf("The amount owed for the third order of ping-pong balls is: %d cents\n", order3_owed);

    return 0;

}

/* Returns the total purchase price (in cents) for an order of ping-pong balls, 
 * given a quantity and a tax rate (between 0 and 1), at a unit price of 20 cents.
 * Volume discounts are applied before tax:
 * - No discount is applied to orders of less than 100 units
 * - Orders of at least 100 units and less than 500 units are discounted 15%
 * - Orders of 500 or more are discounted 35%
 */
long calculate_total(int quantity, double tax_rate) {
    
    double discount;
    if (quantity < 100) {
        discount = 0.0;
    }
    else if (quantity < 500) {
        discount = 0.15;
    }
    else { // Quantity >= 500
        discount = 0.35;
    }

    return lround(quantity * 20 * (1 - discount) * (1 + tax_rate));

}