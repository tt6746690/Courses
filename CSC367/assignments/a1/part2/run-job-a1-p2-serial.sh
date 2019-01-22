#!/bin/bash

#First load all related modules.  
#You can put the below two lines in a batch file. 
#But remember the modules might get unloaded so check you loaded modules frequently. 
module load gcc/7.3.0

#Compile your code
rm -f *.out
make clean
make


if [ -e ./_data_0 ]
then
	#Schedule your jobs with sbatch
	sbatch job-a1-p2-serial.sh
else
	#Generate the data
	jid1=$(sbatch job-a1-p2-datagen.sh | cut -f 4 -d' ')
	echo "jid1=$jid1"
	#Schedule your jobs with sbatch
	sbatch --dependency=afterany:$jid1 job-a1-p2-serial.sh
fi





