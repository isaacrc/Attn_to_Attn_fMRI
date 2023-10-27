#!/usr/bin/env bash

# Run from BIDS code/preprocessing directory: sbatch slurm_mriqc.sh

# Name of job?
#SBATCH --job-name=bpress_all

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23/logs/activation/act_all-%A_%a.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 12:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

# Update with your email 
#SBATCH --mail-user=isaacrc@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --array=1-2
#printf -v roi $SLURM_ARRAY_TASK_ID
#echo "$SLURM_ARRAY_TASK_ID"

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
cd /jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23/activations
chmod +x roi_bpress_2023-6-1.py
./roi_bpress_2023-6-1.py 2


echo "Finished"
date
