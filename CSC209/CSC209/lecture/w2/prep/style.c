#include <stdio.h>

int main() {
    /* This program calculates the result 6 months of compounded interest on a
       principal investment of $500, at an interest rate of 4% annually,
       compounded monthly. */

    // Initial investment
    int principal = 500;  // comment on the same line

    // Initialize calculation variables
    double interest_rate, multiplier, compound_interest;

    // Result of 6 months of compounded interest
    double six_months_compounded;

    // Initial value
    interest_rate = 0.04;

    // The change after each month
    multiplier = 1 + (interest_rate / 12);

    printf("%f", multiplier);

    // Raise the multiplier to the power 6 (number of months)
    compound_interest = multiplier * multiplier * multiplier * multiplier *
        multiplier * multiplier;

    // Compute the final result
    six_months_compounded = principal * compound_interest;

    return 0;
}
