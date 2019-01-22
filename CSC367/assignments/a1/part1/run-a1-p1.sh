#!/bin/bash

#First load all related modules.  
#You can put the below two lines in a batch file. 
#But remember the modules might get unloaded so check you loaded modules frequently. 
module load anaconda3/5.2.0
module load gcc/7.3.0

#Compile your code
make clean
rm -f *.out
make

#Schedule your jobs with sbatch
sbatch job-a1-p1.sh