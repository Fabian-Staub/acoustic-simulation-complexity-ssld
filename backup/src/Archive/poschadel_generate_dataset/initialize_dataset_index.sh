#!/bin/bash
#SBATCH --job-name=initialize_dataset_index
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --output=logs/initialize_dataset_index.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/initialize_dataset_index.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python initialize_dataset_index.py