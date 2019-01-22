#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --job-name a1_p2_serial
#SBATCH --output=a1_p2_serial_%j.out

# You can uncomment any of the following lines to run your
# code with the given data. This script assumes 
# data-generation is completed sucessfully.

./part2 _data_0
#./part2 _data_1
#./part2 _data_2
#./part2 _data_3 
#
