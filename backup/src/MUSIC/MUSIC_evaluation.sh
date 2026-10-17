#!/bin/bash

#SBATCH --job-name=eval_model_music

#SBATCH --time=20:00:00
#SBATCH --output=checkpoints/evaluation_results/logs/eval_MUSIC.out
#SBATCH --error=checkpoints/evaluation_results/logs/eval_MUSIC.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 MUSIC_evaluation_3.py \
  --checkpoint-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_Direct/hopiv_logmel_128 \
  --arni-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/poschadel/hopiv_logmel_128 \
  --motus-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/hopiv_logmel_128 \
  --ivan-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/ivan_em32/poschadel/hopiv_logmel_128 \
  --output /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions_MUSIC.csv \
  --num-bands 128 \
  --workers "$SLURM_CPUS_PER_TASK" \
  #--test-mode