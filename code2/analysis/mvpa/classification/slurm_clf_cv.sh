#!/usr/bin/env bash

# Run from BIDS code/preprocessing directory: sbatch slurm_mriqc.sh

# Name of job?
#SBATCH --job-name=clf_cv13

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23/logs/classification/clf_all-%A_%a.log'

#SBATCH --ntasks=1

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 16:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=16

# Update with your email 
#SBATCH --mail-user=isaacrc@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# roi...?
#SBATCH --array=1-8
printf -v win $SLURM_ARRAY_TASK_ID
echo "$SLURM_ARRAY_TASK_ID"

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
date

#Run script located in directory ##
export PYTHONUNBUFFERED=1

# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
cd /jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23/classification

## leggo ##
python cv13_classify-2023-05-15-roc-spatial.py $win



echo "Finished"
date
