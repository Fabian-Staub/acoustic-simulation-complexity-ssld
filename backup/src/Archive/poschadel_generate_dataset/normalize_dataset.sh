#!/bin/bash
#SBATCH --job-name=normalize_dataset
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

#SBATCH --output=logs/normalize_dataset.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/normalize_dataset.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python normalize_dataset.py \
    --input-folder /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/hopiv_accdoa_mel128_TIMIT_babble_full/pt \
    --stats-json ../../../data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/hopiv_accdoa_mel128_TIMIT_babble_full/normalization_stats.json \
    --stats-file-percent 70 \
    --mode apply-only \
    #--mode calculate-and-apply
    #--mode apply-only