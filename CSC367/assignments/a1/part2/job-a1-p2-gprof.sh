#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:10:00
#SBATCH --job-name a1_p2_gprof
#SBATCH --output=a1_p2_gprof_%j.out

module load gcc/7.3.0

./part2
gprof ./part2 gmon.out > analysis-gprof.out
