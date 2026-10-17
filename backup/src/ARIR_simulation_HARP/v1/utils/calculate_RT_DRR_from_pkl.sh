#!/bin/bash
#SBATCH --job-name=calculate_RT_DRR_from_pkl
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

#SBATCH --output=calculate_RT_DRR_from_pkl_%A.out   # %A=JobID, %a=Array index
#SBATCH --error=calculate_RT_DRR_from_pkl_%A.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python calculate_RT_DRR_from_pkl.py \
"/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_Direct/RIRs"