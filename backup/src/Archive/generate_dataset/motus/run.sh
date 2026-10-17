#!/bin/bash
#SBATCH --job-name=process_measurements
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

#SBATCH --output=job_process_measurements.out   # %A=JobID, %a=Array index
#SBATCH --error=job_process_measurements.err 

module purge
module load mamba
source activate master_thesis_music_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 truncate+rt60.py