#!/bin/bash
#SBATCH --partition=main-redhat
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
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

PROJECT_DIR="/home/mr2320/malinois_inference_minimal"
DATA_DIR="${PROJECT_DIR}/data"
MODEL="${PROJECT_DIR}/model/sei.pth"

case "${DATASET}" in
    K562)
        INPUT_CSV="${DATA_DIR}/K562_processed.csv"
        OUTPUT_PREFIX="MPRAVarDB_K562"
        ;;

    HepG2)
        INPUT_CSV="${DATA_DIR}/HepG2_processed.csv"
        OUTPUT_PREFIX="MPRAVarDB_HepG2"
        ;;

    SKNSH)
        INPUT_CSV="${DATA_DIR}/SKNSH_processed.csv"
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

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${DATA_DIR}"

cd "${PROJECT_DIR}"

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Dataset:            ${DATASET}"
echo "Sequence type:      ${SEQ_TYPE}"
echo "Sequence column:    ${SEQ_COL}"
echo "Input CSV:          ${INPUT_CSV}"
echo "Output prefix:      ${OUTPUT_PREFIX}_${SEQ_TYPE}"
echo "Running on host:    $(hostname)"
echo "Start time:         $(date)"
echo

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate malinois_inference

python predict_sei.py \
    --input_csv "${INPUT_CSV}" \
    --delimiter "," \
    --seq_col "${SEQ_COL}" \
    --pretrained "${MODEL}" \
    --batch_size 512 \
    --out_npy "${DATA_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_21k.npy" \
    --out_csv "${DATA_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_21k.csv" \
    --out_seq_classes "${DATA_DIR}/${OUTPUT_PREFIX}_${SEQ_TYPE}_seq_classes.npy"

echo
echo "Finished successfully: $(date)"