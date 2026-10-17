#!/bin/bash
#SBATCH --job-name=generate_dataset
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --output=logs/generate_dataset_R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder1-Late.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/generate_dataset_R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder1-Late.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


python preprocess_hopiv_7.py \
  --mode train \
  --split 70/20/10 \
  --base-dir /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder1-Late/RIRs \
  --output-folder ../hopiv_logmel_128 \
  --write-wavs testset \
  --calculate-distribution \
  --rooms-per-pt 10 \
  #--nrooms 10