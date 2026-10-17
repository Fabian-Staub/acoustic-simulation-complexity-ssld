#!/bin/bash

#SBATCH --job-name=R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_stft_05-droput

#SBATCH --time=10:00:00
#SBATCH --output=trained_models/train_model_%x.out
#SBATCH --error=trained_models/train_model_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 train_crossval_5_stft.py \
  --data_path /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late/harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_stft.pt
