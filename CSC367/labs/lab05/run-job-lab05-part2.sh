#!/bin/bash

module load gcc/7.3.0
module load openmpi

rm -f *.txt
rm -f *.out
make 

sbatch job-lab05-part2.sh
