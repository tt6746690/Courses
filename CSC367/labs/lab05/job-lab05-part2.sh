#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:05:00
#SBATCH --job-name lab05_part2
#SBATCH --output=lab05_part2_%j.out


module load gcc/7.3.0
module load openmpi/3.1.1

#number of mpi tasks
ntasks=8

mpirun -np $ntasks ./transpose-alltoall


