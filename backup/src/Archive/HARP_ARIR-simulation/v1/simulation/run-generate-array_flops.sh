#!/bin/bash
#SBATCH --job-name=generate_array
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0

#SBATCH --output=logs/test_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=logs/test_%a.err 

module purge
module load mamba
source activate master_thesis_pyroom_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python3 Generate_rtProfiles_non_shoebox_array_32_flops_poschadel.py \
  --sample_rate 24000 \
  --em32_directivity_path "Eigenmike_em32_IRs.npz" \
  --amp_threshold 15 \
  --num_rooms 1 \
  --num_positions 1 \
  --num_sources_per_receiver 1 \
  --ambi_order 3 \
  --output_path "../../../data/harp/test_flops" \
  --array_type em32 \
  --ism_order 12 \
  --export_mode ism \
  --shoebox \
  --max_displacement_corners 0.0 \
  --no-use_rand_ism \
  --max_rand_disp 0 \
  --no-save_plots \
  --hardware_flops