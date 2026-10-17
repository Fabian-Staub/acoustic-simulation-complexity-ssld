#!/bin/bash
#SBATCH --job-name=generate_dataset
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0

#SBATCH --output=logs/generate_dataset_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/generate_dataset_%a.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


BASE_DIR=/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30

python preprocess_harp_scm_13.py \
    --order 3 \
    --base-dir "$BASE_DIR" \
    --output-folder "$BASE_DIR/hopiv_accdoa_mel128_TIMIT_babble_test" \
    --random-rotation \
    --write-wavs \
    --plot-first-sample \
    --nrooms 10 \