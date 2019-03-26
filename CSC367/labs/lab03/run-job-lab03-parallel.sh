#!/bin/bash

module load gcc/7.3.0

rm -f *.txt
rm -f *.out
make 

sbatch job-lab03-parallel.sh
