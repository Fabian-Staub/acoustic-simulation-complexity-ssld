#!/bin/bash
#SBATCH --job-name=preprocess_array
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-9

#SBATCH --output=job_preprocess_array1_%a.out   # %A=JobID, %a=Array index
#SBATCH --error=job_preprocess_array1_%a.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python preprocess_harp_scm_randRot_array_5.py \
  --base-dir ../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late \
  --dataset-name harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats \
  --speech-folder ../../../data/audio_exports_Harvard_Sets \
  --noise-folder ../../../data/noises \
  --order 4 \
  --fs 24000 \
  --segment-len-s 1 \
  --min-voice 1 \
  --max-voice 2 \
  --max-noise 0 \
  --noise-volume-ratio 0 \
  --num-repeats 10 \
  --random-rotation \
  --highpass \
  --num-scms 10 \
  --freq-range 2000 8000 \
  #--rir-noise-db-range 60 70 \