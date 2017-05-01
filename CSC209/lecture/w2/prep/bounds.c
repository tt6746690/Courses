int main() {
    // This code doesn't print anything.
    // To see what it is doing, run it in the memory visualizer.
    
    int A[3] = {13, 55, 20};
    int B[3] = {1, 2, 3};

    
    int in_bounds = A[1];

    // This line accesses memory that is not part of array A.
    // It might cause an error and stop the execution (crash) 
    // or it might appear to work and assign a random value to out_of_bounds.
    // Try changing it to int out_of_bounds = A[3000];
    int out_of_bounds = A[3];
    

    // We can assign to something out of bounds as well. It is wrong
    // but may or may not cause a visible error.
    A[5] = 999;

    return 0;
}