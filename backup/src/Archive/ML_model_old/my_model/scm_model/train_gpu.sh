#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=job_train_mlp_2.out
#SBATCH --error=job_train_mlp_2.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env

python3 train_mlp.py
