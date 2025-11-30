#!/usr/bin/env bash
set -euo pipefail

# Simple helper to evaluate k-NN on the digit strokes.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python}

DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/../../digits_3d/training_data"}
SEQ_LEN=${SEQ_LEN:-128}
K_LIST=${K_LIST:-"1,2,3,4,5,6,7,8,9"}
FOLDS=${FOLDS:-5}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/results.json"}

echo "Running k-NN..."
echo "DATA_DIR=${DATA_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --data_dir "${DATA_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --k_list "${K_LIST}" \
  --folds "${FOLDS}" \
  --seed "${SEED}" \
  --output "${OUTPUT}"

echo "Done."
