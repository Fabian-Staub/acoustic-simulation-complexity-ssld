#!/bin/bash

#SBATCH --job-name=train_model

#SBATCH --time=20:00:00
#SBATCH --output=checkpoints/logs/train_shoeBox_SH_ISMOrder1.out
#SBATCH --error=checkpoints/logs/train_shoeBox_SH_ISMOrder1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 train_10.py \
    --dataset-root "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder1/hopiv_logmel_128" \
    --checkpoint-path "checkpoints/shoeBox_SH_ISMOrder1.pth" \
    --batch-size 800 \
    --learning-rate 1e-3 \
    --weight-decay 3e-4 \
    --patience 20 \
    --min-delta 2e-4 \
    --min-epochs 1 \
    --lr-factor 0.5 \
    --lr-patience 5 \
    --lr-min-delta 2e-4
