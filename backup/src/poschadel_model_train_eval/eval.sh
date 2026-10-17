#!/bin/bash

#SBATCH --job-name=eval_model

#SBATCH --time=00:30:00
#SBATCH --output=checkpoints/evaluation_results/logs/eval_shoeBox_em32_ISMOrder1.out
#SBATCH --error=checkpoints/evaluation_results/logs/eval_shoeBox_em32_ISMOrder1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1


python eval_5.py \
    --checkpoint /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/shoeBox_em32_ISMOrder1.pth \
    --train-module train_10 \
    --arni-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/poschadel/hopiv_logmel_128 \
    --motus-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/hopiv_logmel_128 \
    --ivan-dataset /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/ivan_em32/poschadel/hopiv_logmel_128 \
    --batch-size 64 \
    --num-workers 6 \
    --output-dir checkpoints/evaluation_results