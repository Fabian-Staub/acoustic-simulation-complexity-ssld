#!/bin/bash
#SBATCH --job-name=generate_array
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-9

#SBATCH --output=logs/generate_shoeBox_SH_ISMOrder1-Late_array_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/generate_shoeBox_SH_ISMOrder1-Late_array_%a.err 

module purge
module load mamba
source activate master_thesis_pyroom_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 Generate_rtProfiles_non_shoebox_array_32_poschadel.py \
  --sample_rate 24000 \
  --em32_directivity_path "Eigenmike_em32_IRs.npz" \
  --amp_threshold 15 \
  --num_rooms 2000 \
  --num_positions 8 \
  --num_sources_per_receiver 3 \
  --ambi_order 3 \
  --output_path "../../../data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder1-Late/RIRs" \
  --array_type ambisonic \
  --ism_order 1 \
  --export_mode ism_late \
  --shoebox \
  --max_displacement_corners 0.0 \
  --no-use_rand_ism \
  --max_rand_disp 0 \
  --no-save_plots