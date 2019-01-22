#!/bin/bash

#First load all related modules.  
#You can put the below two lines in a batch file. 
#But remember the modules might get unloaded so check you loaded modules frequently. 
module load gcc/7.3.0

#Generate data 
make part2data

#Compile your code
gcc -c -g data.c -o data.o -pg
# You may replace this part to profile parallel code or parallel-opt code
gcc -c -g part2.c -o part2.o -pg
gcc -o part2 data.o part2.o -pg


#Schedule your jobs with sbatch
sbatch  job-a1-p2-gprof.sh




