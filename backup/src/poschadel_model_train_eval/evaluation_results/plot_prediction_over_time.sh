#!/bin/bash
#SBATCH --job-name=plot_prediction_over_time
#SBATCH --time=00:10:00
#SBATCH --output=plot_prediction_over_time.out
#SBATCH --error=plot_prediction_over_time.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python plot_prediction_over_time_2.py \
  --dataset \
      /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results//predictions/predictions_MUSIC.csv \
      /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results//predictions/predictions_shoeBox_em32_Direct.csv \
      /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results//predictions/predictions_shoeBox_em32_ISMOrder2.csv \
      /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results//predictions/predictions_shoeBox_em32_ISMOrder12.csv \
  --labels "MUSIC" "Mic: EM | ISM-Order: 0 | Late: no" "Mic: EM | ISM-Order: 2 | Late: no" "Mic: EM | ISM-Order: 12 | Late: no" \
  --sample-id R0002_P65 \
  --prediction-dataset arni-dataset \
  --output plot_prediction_over_time_1.pdf