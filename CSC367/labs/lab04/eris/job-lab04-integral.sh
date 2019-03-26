#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --job-name lab04_integral
#SBATCH --output=lab04_integral_%j.out

Nthreads=8
export OMP_NUM_THREADS=$Nthreads

./integral-seq
./integral-omp

