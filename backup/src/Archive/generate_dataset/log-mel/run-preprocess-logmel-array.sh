#!/bin/bash
#SBATCH --job-name=preprocess_array
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --output=job_preprocess_array_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=job_preprocess_array_%a.err 

module purge
module load mamba
source activate master_thesis_env

python3 preprocess_harp_logmel_array_3.py