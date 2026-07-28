#!/bin/bash
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch --job-name=<dataset>_<ref|alt> $0 <dataset> <ref|alt>"
    echo "Datasets: K562, HepG2, SKNSH"
    exit 1
fi

DATASET="$1"
SEQ_TYPE="$2"

# Directory containing predict_sei.py and model/sei.pth.
# Change this to the actual SEI code directory.
CODE_DIR="/scratch/mr2320/run_sei"

# Input data are stored here.
INPUT_DIR="/home/mr2320/malinois_inference_minimal/data"

# Outputs will be written here.
OUTPUT_DIR="/home/mr2320/malinois_inference_minimal/data"

PREDICT_SCRIPT="${CODE_DIR}/predict_sei.py"
PRETRAINED_MODEL="${CODE_DIR}/model/sei.pth"

case "${DATASET}" in
    K562)
        INPUT_FILE="${INPUT_DIR}/K562_processed.csv"
        OUTPUT_PREFIX="MPRAVarDB_K562"
        ;;

    HepG2)
        INPUT_FILE="${INPUT_DIR}/HepG2_processed.csv"
        OUTPUT_PREFIX="MPRAVarDB_HepG2"
        ;;

    SKNSH)
        INPUT_FILE="${INPUT_DIR}/SKNSH_processed.csv"
        OUTPUT_PREFIX="MPRAVarDB_SKNSH"
        ;;

    *)
        echo "Error: unsupported dataset '${DATASET}'"
        echo "Supported datasets: K562, HepG2, SKNSH"
        exit 1
        ;;
esac

case "${SEQ_TYPE}" in
    ref)
        SEQ_COL="ref_seq"
        ;;

    alt)
        SEQ_COL="alt_seq"
        ;;

    *)
        echo "Error: sequence type must be 'ref' or 'alt'"
        exit 1
        ;;
esac

# Verify required files before starting the prediction.
if [[ ! -f "${PREDICT_SCRIPT}" ]]; then
    echo "Error: prediction script not found:"
    echo "${PREDICT_SCRIPT}"
    exit 1
fi

if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "Error: pretrained model not found:"
    echo "${PRETRAINED_MODEL}"
    exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found:"
    echo "${INPUT_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate malinois_inference

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Job name:        ${SLURM_JOB_NAME}"
echo "Dataset:         ${DATASET}"
echo "Sequence type:   ${SEQ_TYPE}"
echo "Sequence column: ${SEQ_COL}"
echo "Code directory:  ${CODE_DIR}"
echo "Input file:      ${INPUT_FILE}"
echo "Model:           ${PRETRAINED_MODEL}"
echo "Output directory:${OUTPUT_DIR}"
echo "Conda env:       ${CONDA_DEFAULT_ENV}"
echo "Python:          $(which python)"
echo "Host:            $(hostname)"
echo "Start time:      $(date)"
echo

python "${PREDICT_SCRIPT}" \
    --input_csv "${INPUT_FILE}" \
    --delimiter "," \
    --seq_col "${SEQ_COL}" \
    --pretrained "${PRETRAINED_MODEL}" \
    --batch_size 512 \
    --out_npy "${OUTPUT_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_21k.npy" \
    --out_csv "${OUTPUT_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_21k.csv" \
    --out_seq_classes "${OUTPUT_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_seq_classes.npy"

echo
echo "Finished successfully: $(date)"