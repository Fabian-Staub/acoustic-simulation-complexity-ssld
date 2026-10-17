#!/bin/bash

#SBATCH --job-name=R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats_best-thresh

#SBATCH --time=02:00:00
#SBATCH --output=eval_models/eval_model_%x.out
#SBATCH --error=eval_models/eval_model_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python eval_crossval_3.py \
  --val_path ../../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late/harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats.pt \
  --arni_path ../../../../data/arni_em32/sh_wavs/arni_dataset_2-8kHz_10-SCM_2-src_3-repeats.pt \
  --ivan_path ../../../../data/SummerAaltoMeasurements/processed_data/sh_wavs/ivan_dataset_2-8kHz_10-SCM_2-src_3-repeats.pt \
  --find_best_accdoa_thresh
