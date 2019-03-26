#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --job-name lab04_primes
#SBATCH --output=lab04_primes_%j.out

Nthreads=8
export OMP_NUM_THREADS=$Nthreads

# The method will count the total number of prime numbers between 2 and N
N=50000

# Chunk size for statical scheduling and dynamical scheduling used for openmp codes 
# chunk_size=5

echo "---------------seq--------------"
./primes-seq $N

echo "---------------default----------"
./primes-default $N

echo "---------------static-----------"


for chunk_size in 5 10 20 40 80 160
do
   echo "static chunk_size=$chunk_size"
    ./primes-static $N $chunk_size
done

for chunk_size in 5 10 20 40 80 160
do
   echo "dynamic chunk_size=$chunk_size"
    ./primes-static $N $chunk_size
done

