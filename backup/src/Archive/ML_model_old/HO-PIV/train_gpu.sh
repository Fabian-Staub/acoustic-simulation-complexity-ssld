#!/bin/bash

#SBATCH --job-name=test

#SBATCH --time=02:00:00
#SBATCH --output=train_model_%x.out
#SBATCH --error=train_model_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 train.py
