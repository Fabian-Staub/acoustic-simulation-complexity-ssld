#!/bin/bash
#SBATCH --job-name=accdoa_resnet
#SBATCH --output=job_accdoa_resnet.out
#SBATCH --error=accdoa_resnet.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env

python3 ai_model_3.py
