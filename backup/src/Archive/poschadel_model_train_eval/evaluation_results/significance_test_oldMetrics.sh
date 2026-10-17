#!/bin/bash
#SBATCH --job-name=significance_test
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --output=checkpoints/evaluation_results/significance_test/logs/significance_test.out   # %A=JobID, %a=Array index
#SBATCH --error=checkpoints/evaluation_results/significance_test/logs/significance_test.err 

module purge
module load mamba
source activate master_thesis_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 significance_test_8_oldMetrics.py \
    --csvs \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_Direct.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_Direct.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_Direct-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_Direct-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_ISMOrder2.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_ISMOrder2.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_ISMOrder2-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_ISMOrder2-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_SH_ISMOrder12-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_em32_ISMOrder12-Late.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_non-shoeBox_disp-corners-05_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_non-shoeBox_disp-corners-1_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_non-shoeBox_disp-corners-3_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12.csv \
    /scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_MUSIC.csv \
    --output-dir checkpoints/evaluation_results/significance_test \
    --workers "$SLURM_CPUS_PER_TASK"
