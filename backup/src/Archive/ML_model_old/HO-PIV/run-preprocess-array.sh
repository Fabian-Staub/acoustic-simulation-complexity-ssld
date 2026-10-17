#!/bin/bash
#SBATCH --job-name=preprocess_array
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-9

#SBATCH --output=job_preprocess_array_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=job_preprocess_array_%a.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python build_hopiv_timit_dataset.py