#!/usr/bin/env bash
set -euo pipefail

# Run SVM baseline with kernel ablation.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python}

DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/../../digits_3d/training_data"}
SEQ_LEN=${SEQ_LEN:-128}
KERNELS=${KERNELS:-"linear,rbf,poly,sigmoid"}
C_VAL=${C_VAL:-1.0}
GAMMA=${GAMMA:-"scale"}
DEGREE=${DEGREE:-3}
VAL_SPLIT=${VAL_SPLIT:-0.2}
SEED=${SEED:-0}
OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/svm_results.json"}

echo "Running SVM..."
echo "DATA_DIR=${DATA_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --data_dir "${DATA_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --kernels "${KERNELS}" \
  --C "${C_VAL}" \
  --gamma "${GAMMA}" \
  --degree "${DEGREE}" \
  --val_split "${VAL_SPLIT}" \
  --seed "${SEED}" \
  --output "${OUTPUT}"

echo "Done."
