#!/usr/bin/env bash

set -euo pipefail

# Folder containing this script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${PROJECT_DIR}/preprocess_harp_scm_11.py"

# Optional number of rooms.
# Comment this line out to process all rooms available to the Python script.
#NROOMS=1000

NUM_TASKS=1
ARRAY_RANGE="0-$((NUM_TASKS - 1))"

# Dataset and output folders
BASE_DIR="${PROJECT_DIR}/../../../data/ivan_em32/processed_data/sh_wavs"
OUTPUT_FOLDER="${BASE_DIR}/hopiv_accdoa_mel128_TIMIT_babble"

PT_FOLDER="${OUTPUT_FOLDER}/pt"
CSV_FOLDER="${OUTPUT_FOLDER}/csv"

cd "${PROJECT_DIR}"
mkdir -p logs

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

if ! command -v sbatch >/dev/null 2>&1; then
    echo "The sbatch command is not available." >&2
    exit 1
fi

if (( NUM_TASKS <= 0 )); then
    echo "NUM_TASKS must be greater than zero." >&2
    exit 1
fi

if [[ -n "${NROOMS+x}" ]] && (( NROOMS <= 0 )); then
    echo "NROOMS must be greater than zero when set." >&2
    exit 1
fi

echo "Submitting raw-feature generation pipeline"
echo "Project directory: ${PROJECT_DIR}"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Base directory:    ${BASE_DIR}"
echo "Output folder:     ${OUTPUT_FOLDER}"
echo "Partition:         Slurm default"
echo "Array tasks:       ${NUM_TASKS}"

if [[ -n "${NROOMS+x}" ]]; then
    echo "Rooms:             ${NROOMS}"
else
    echo "Rooms:             Python default"
fi

echo

# ------------------------------------------------------------
# Generation array
# ------------------------------------------------------------

# Store an empty value when NROOMS is unset.
NROOMS_VALUE="${NROOMS-}"

GENERATE_JOB_ID="$(
    sbatch \
        --parsable \
        --array="${ARRAY_RANGE}" \
        --job-name=harp_generate_raw \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=02:00:00 \
        --output="${PROJECT_DIR}/logs/generate_arni_%a.out" \
        --error="${PROJECT_DIR}/logs/generate_arni_%a.err" \
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

ARGS=(
    --order 3
    --task-id \"\${SLURM_ARRAY_TASK_ID}\"
    --num-tasks '${NUM_TASKS}'
    --base-dir '${BASE_DIR}'
    --output-folder '${OUTPUT_FOLDER}'
)

if [[ -n '${NROOMS_VALUE}' ]]; then
    ARGS+=(--nrooms '${NROOMS_VALUE}')
fi

python -u '${PYTHON_SCRIPT}' \"\${ARGS[@]}\"
"
)"

# Some Slurm configurations append a cluster name:
# 123456;cluster-name
GENERATE_JOB_ID="${GENERATE_JOB_ID%%;*}"

if [[ -z "${GENERATE_JOB_ID}" ]]; then
    echo "Failed to obtain generation job ID." >&2
    exit 1
fi

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo "Generation array job: ${GENERATE_JOB_ID}"
echo
echo "Submitted raw-feature generation pipeline:"
echo "  Generation array: ${GENERATE_JOB_ID}"
echo
echo "Using dataset:"
echo "  ${BASE_DIR}"
echo
echo "Expected outputs:"
echo "  PT files:"
echo "    ${PT_FOLDER}/"
echo
echo "  CSV files:"
echo "    ${CSV_FOLDER}/"
echo
echo "Monitor generation jobs:"
echo "  squeue -j ${GENERATE_JOB_ID}"
echo
echo "Monitor all your jobs:"
echo "  squeue -u \"${USER}\""
echo
echo "Inspect the array job:"
echo "  scontrol show job ${GENERATE_JOB_ID}"
echo
echo "Cancel this generation pipeline:"
echo "  scancel ${GENERATE_JOB_ID}"
echo
echo "Cancel all your Slurm jobs:"
echo "  scancel -u \"${USER}\""
echo
echo "Logs:"
echo "  ${PROJECT_DIR}/logs/"
