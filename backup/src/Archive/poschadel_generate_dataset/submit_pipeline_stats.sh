#!/usr/bin/env bash

set -euo pipefail

# Folder containing this script and preprocess_harp_scm_6.py
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${PROJECT_DIR}/preprocess_harp_scm_8.py"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Directory containing:
#   Generated_HOA_SRIR_data.csv
#   RIRs/
BASE_DIR="${PROJECT_DIR}/../../../data/harp/R4500_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30"

# Directory in which partial and merged statistics are written
OUTPUT_FOLDER="${BASE_DIR}/hopiv_accdoa_mel128_TIMIT"

NUM_TASKS=10
NROOMS=700
ARRAY_RANGE="0-$((NUM_TASKS - 1))"

PARTIAL_STATS_FOLDER="${OUTPUT_FOLDER}/partial_stats"
STATS_FILE="${OUTPUT_FOLDER}/feature_normalization_stats.json"
LOG_FOLDER="${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}"

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Missing Python script: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Missing base directory: ${BASE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${BASE_DIR}/Generated_HOA_SRIR_data.csv" ]]; then
    echo "Missing RIR metadata CSV:" >&2
    echo "  ${BASE_DIR}/Generated_HOA_SRIR_data.csv" >&2
    exit 1
fi

if [[ ! -d "${BASE_DIR}/RIRs" ]]; then
    echo "Missing RIR folder:" >&2
    echo "  ${BASE_DIR}/RIRs" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "The sbatch command is not available." >&2
    exit 1
fi

if (( NUM_TASKS <= 0 )); then
    echo "NUM_TASKS must be greater than zero." >&2
    exit 1
fi

if (( NROOMS <= 0 )); then
    echo "NROOMS must be greater than zero." >&2
    exit 1
fi

echo "Submitting feature-statistics pipeline"
echo "Project directory:        ${PROJECT_DIR}"
echo "Python script:            ${PYTHON_SCRIPT}"
echo "Base directory:           ${BASE_DIR}"
echo "Output folder:            ${OUTPUT_FOLDER}"
echo "Partial statistics:       ${PARTIAL_STATS_FOLDER}"
echo "Merged statistics file:   ${STATS_FILE}"
echo "Array tasks:              ${NUM_TASKS}"
echo "Rooms:                    ${NROOMS}"
echo

# ------------------------------------------------------------
# Partial-statistics array
# ------------------------------------------------------------

STATS_JOB_ID="$(
    sbatch \
        --parsable \
        --array="${ARRAY_RANGE}" \
        --job-name=harp_stats \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=8G \
        --time=02:00:00 \
        --output="${LOG_FOLDER}/stats_%A_%a.out" \
        --error="${LOG_FOLDER}/stats_%A_%a.err" \
        --wrap="
set -euo pipefail

cd '${PROJECT_DIR}'

module purge
module load mamba
source activate master_thesis_env

export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"
export MPLBACKEND=Agg

export OMP_NUM_THREADS=\"\${SLURM_CPUS_PER_TASK}\"
export MKL_NUM_THREADS=\"\${SLURM_CPUS_PER_TASK}\"
export OPENBLAS_NUM_THREADS=\"\${SLURM_CPUS_PER_TASK}\"
export NUMEXPR_NUM_THREADS=\"\${SLURM_CPUS_PER_TASK}\"

python -u '${PYTHON_SCRIPT}' \\
    --mode stats \\
    --task-id \"\${SLURM_ARRAY_TASK_ID}\" \\
    --num-tasks '${NUM_TASKS}' \\
    --nrooms '${NROOMS}' \\
    --base-dir '${BASE_DIR}' \\
    --output-folder '${OUTPUT_FOLDER}'
"
)"

# Some Slurm configurations return:
#   123456;cluster-name
STATS_JOB_ID="${STATS_JOB_ID%%;*}"

if [[ -z "${STATS_JOB_ID}" ]]; then
    echo "Failed to obtain statistics-array job ID." >&2
    exit 1
fi

echo "Statistics array job: ${STATS_JOB_ID}"

# ------------------------------------------------------------
# Merge partial statistics
# ------------------------------------------------------------

MERGE_STATS_JOB_ID="$(
    sbatch \
        --parsable \
        --dependency="afterok:${STATS_JOB_ID}" \
        --job-name=harp_merge_stats \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:30:00 \
        --output="${LOG_FOLDER}/merge_stats_%j.out" \
        --error="${LOG_FOLDER}/merge_stats_%j.err" \
        --wrap="
set -euo pipefail

cd '${PROJECT_DIR}'

module purge
module load mamba
source activate master_thesis_env

export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"
export MPLBACKEND=Agg

python -u '${PYTHON_SCRIPT}' \\
    --mode merge-stats \\
    --expected-tasks '${NUM_TASKS}' \\
    --stats-file '${STATS_FILE}' \\
    --base-dir '${BASE_DIR}' \\
    --output-folder '${OUTPUT_FOLDER}'
"
)"

MERGE_STATS_JOB_ID="${MERGE_STATS_JOB_ID%%;*}"

if [[ -z "${MERGE_STATS_JOB_ID}" ]]; then
    echo "Failed to obtain statistics-merge job ID." >&2
    exit 1
fi

echo "Statistics merge job: ${MERGE_STATS_JOB_ID}"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "Submitted statistics pipeline:"
echo "  Statistics array: ${STATS_JOB_ID}"
echo "  Statistics merge: ${MERGE_STATS_JOB_ID}"
echo
echo "Dependency chain:"
echo "  ${STATS_JOB_ID} -> ${MERGE_STATS_JOB_ID}"
echo
echo "Expected partial files:"
echo "  ${PARTIAL_STATS_FOLDER}/stats_task_0000.json"
echo "  ..."
printf "  %s/stats_task_%04d.json\n" \
    "${PARTIAL_STATS_FOLDER}" \
    "$((NUM_TASKS - 1))"
echo
echo "Expected merged file:"
echo "  ${STATS_FILE}"
echo
echo "Monitor pipeline jobs:"
echo "  squeue -j ${STATS_JOB_ID},${MERGE_STATS_JOB_ID}"
echo
echo "Monitor all your jobs:"
echo "  squeue -u \"${USER}\""
echo
echo "Inspect merge dependency:"
echo "  scontrol show job ${MERGE_STATS_JOB_ID}"
echo
echo "Cancel this pipeline:"
echo "  scancel ${STATS_JOB_ID} ${MERGE_STATS_JOB_ID}"
echo
echo "Logs:"
echo "  ${LOG_FOLDER}/"
