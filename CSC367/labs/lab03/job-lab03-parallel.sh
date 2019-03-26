#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --job-name lab03_parallel
#SBATCH --output=lab03_parallel_%j.out

hash_table_size=1000
keys_per_bucket=5
operations_count=50000
write_precentage=30
threads_count=8

exec="./test-perf-parallel"

$exec $hash_table_size $keys_per_bucket $operations_count $write_precentage $threads_count


