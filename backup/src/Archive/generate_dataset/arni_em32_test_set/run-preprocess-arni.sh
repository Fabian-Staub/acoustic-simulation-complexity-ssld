#!/bin/bash
#SBATCH --job-name=preprocess_arni
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --output=job_preprocess_arni1_scm.out   # %A=JobID, %a=Array index
#SBATCH --error=job_preprocess_arni1_scm.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 preprocess_arni_scm_7.py