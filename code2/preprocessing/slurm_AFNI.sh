#!/bin/bash

# Run from BIDS code/preprocessing directory: sbatch slurm_mriqc.sh

# Name of job?
#SBATCH --job-name=pristine_plumage

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/logs/afni-%A_%a.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 25:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

# Update with your email 
#SBATCH --mail-user=isaacrc@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
date

# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni/2020.10.21
cd /jukebox/graziano/coolCatIsaac/ATM/code/preprocessing
./afni_proc_batch_rest_TEMPLATE.s

echo "Finished"
date
