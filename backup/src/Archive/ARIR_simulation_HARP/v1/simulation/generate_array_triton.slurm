#!/bin/bash -l

#SBATCH --job-name=generate_array
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-0
#SBATCH --constraint=csl
#SBATCH --output=generate_array_%A_%a.out
#SBATCH --error=generate_array_%A_%a.err

set -euo pipefail

module purge
module load mamba
source activate master_thesis_pyroom_env

# ProcessPoolExecutor already uses SLURM_CPUS_PER_TASK workers.  Keep each
# worker's numerical libraries single-threaded to avoid 16 x 16 oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"



srun python3 "Generate_rtProfiles_non_shoebox_array_32_flops_poschadel.py" \
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
