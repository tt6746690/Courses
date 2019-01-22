#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --job-name a1_p2_datagen
#SBATCH --output=a1_p2_datagen_%j.out

./datagen -c 2 -g 5000000 _data_0
./datagen -c 4 -g 5000000 _data_1
./datagen -c 6 -g 5000000 _data_2
./datagen -c 8 -g 5000000 _data_3 

