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
chunk_size=5

./primes-seq $N

./primes-default $N

./primes-static $N $chunk_size

./primes-dynamic $N $chunk_size
