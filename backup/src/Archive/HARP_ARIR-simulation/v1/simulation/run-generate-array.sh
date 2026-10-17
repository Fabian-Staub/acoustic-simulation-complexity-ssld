#!/bin/bash
#SBATCH --job-name=generate_array
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-9

#SBATCH --output=job_generate_array_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=job_generate_array_%a.err 

module purge
module load mamba
source activate master_thesis_env

python3 Generate_mod_array_3.py --output_path "../../../data//harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/" --num_rooms 800 --num_positions 10 --ambi_order 4 --sample_rate 24000 --num_sources_per_receiver 1