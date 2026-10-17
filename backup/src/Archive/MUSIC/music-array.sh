#!/bin/bash
#SBATCH --job-name=eval_music
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --array=0

#SBATCH --output=job_eval_MUSIC_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=job_eval_MUSIC_%a.err 

module purge
module load mamba
source activate master_thesis_music_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 music_12.py