#!/bin/bash
#SBATCH --job-name=preprocess_ivan
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --output=job_preprocess_ivan1_scm.out   # %A=JobID, %a=Array index
#SBATCH --error=job_preprocess_ivan1_scm.err 

module purge
module load mamba
source activate master_thesis_music_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 preprocess_ivan_scm_4.py