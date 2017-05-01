#include <stdio.h>
#include <math.h>

long calculate_total(int quantity, double tax_rate);

int main() {
    int order1_owed = calculate_total(125, 0.10);  //expecting 2338
    int main_quantity = 1000;
    double main_tax = 0.11;
    int order2_owed = calculate_total(main_quantity, main_tax); // expecting 14690
    double discount = 0.5;
    int order3_owed = calculate_total(main_quantity / 100, main_tax * 2);   // expecting 230

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

    double discount = 0.0;
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
