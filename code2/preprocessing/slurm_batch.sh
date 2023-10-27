#!/usr/bin/env bash

# Run from BIDS code/preprocessing directory: sbatch slurm_mriqc.sh

# Name of job?
#SBATCH --job-name=pristine_plumage

#SBATCH --array=1-3

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='logs/test-%A_%a.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 3:00

echo "slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID

printf -v subj "%03d" $SLURM_ARRAY_TASK_ID
printf -v subj1 $SLURM_ARRAY_TASK_ID

echo $subj1
echo "running ish on sub-$subj"
# Print job submission info
date


echo "Finished"
