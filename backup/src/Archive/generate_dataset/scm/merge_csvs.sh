#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --output=job_merge.out   # %A=JobID, %a=Array index
#SBATCH --error=job_merge.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python merge_all_csv_pt.py