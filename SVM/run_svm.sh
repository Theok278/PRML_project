#!/usr/bin/env bash
set -euo pipefail

# Run SVM baseline with kernel ablation.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python3}

DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/../../digits_3d/training_data"}
SEQ_LEN=${SEQ_LEN:-128}
C_VAL=${C_VAL:-5.0}
SEED=${SEED:-0}
OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/svm_results.json"}
K_FOLD=${K_FOLD:-5}

echo "Running SVM..."
echo "DATA_DIR=${DATA_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --data_dir "${DATA_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --C "${C_VAL}" \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  --k_folds "${K_FOLD}"

echo "Done."
