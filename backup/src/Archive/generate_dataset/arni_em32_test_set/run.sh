#!/bin/bash
#SBATCH --job-name=estimate_rt60
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --output=job_estimate_rt60.out   # %A=JobID, %a=Array index
#SBATCH --error=job_estimate_rt60.err 

module purge
module load mamba
source activate master_thesis_music_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 truncate+rt60.py