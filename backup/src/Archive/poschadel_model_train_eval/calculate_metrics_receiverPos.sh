#!/bin/bash
#SBATCH --job-name=initialize_dataset_index
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

#SBATCH --output=test.out   # %A=JobID, %a=Array index
#SBATCH --error=test.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python calculate_metrics_receiverPos.py \
    --predictions /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions_shoeBox_SH_ISMOrder12.csv