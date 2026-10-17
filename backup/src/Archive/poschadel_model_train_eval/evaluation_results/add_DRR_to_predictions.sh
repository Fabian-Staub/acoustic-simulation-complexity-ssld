#!/bin/bash
#SBATCH --job-name=add_DRR_to_predictions
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

#SBATCH --output=add_DRR_to_predictions_%A.out   # %A=JobID, %a=Array index
#SBATCH --error=add_DRR_to_predictions_%A.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python add_DRR_to_predictions.py \
  --predictions-csv /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_MUSIC.csv \
  --checkpoint-metadata /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_Direct/RIRs/02_metadata_T60_DRR.csv \
  --motus-metadata /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/RIRs/02_metadata_T60_DRR.csv \
  --ivan-metadata /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/ivan_em32/poschadel/RIRs/02_metadata_T60_DRR.csv \
  --arni-metadata /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/poschadel/RIRs/02_metadata_T60_DRR.csv \
  --overwrite