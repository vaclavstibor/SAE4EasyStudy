#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-ml-32m-filtered}"

echo "Training ELSA models..."
for embedding_dim in 512 1024 2048; do
    python train_elsa.py --dataset "$DATASET" --embedding_dim "$embedding_dim" --lr 3e-4 --epochs 25 --early_stopping 10 &
done
wait
