#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-ml-32m-filtered}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_CKPT="${SAVE_CKPT:-True}"

RESULTS_DIR="results/${DATASET}"
CHECKPOINTS_DIR="checkpoints/${DATASET}"
LOGS_DIR="logs/${DATASET}"

# Backfill scope: which ELSA input dimensions should be completed.
# Default focuses on the known gap in current results (input_dim=1024).
# Example:
#   TARGET_INPUT_DIMS="1024 2048 4096" bash run_backfill_missing_measurements.sh
TARGET_INPUT_DIMS_STR="${TARGET_INPUT_DIMS:-1024}"
read -r -a TARGET_INPUT_DIMS <<< "${TARGET_INPUT_DIMS_STR}"

# Target SAE grid and canonical long-training setup.
SAE_SCALING_FACTORS=(2 4 8)
SAE_TOPKS=(8 16 32 64)
SAE_EPOCHS="${SAE_EPOCHS:-250}"
SAE_EARLY_STOPPING="${SAE_EARLY_STOPPING:-50}"
SAE_LR="${SAE_LR:-3e-4}"
SAE_L1_COEF="${SAE_L1_COEF:-0.0003}"
SAE_RECON_LOSS="${SAE_RECON_LOSS:-Cosine}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter '${PYTHON_BIN}' not found."
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${CHECKPOINTS_DIR}" "${LOGS_DIR}"

echo "=== BACKFILL MISSING MEASUREMENTS START ==="
echo "Dataset: ${DATASET}"
echo "Target input dims: ${TARGET_INPUT_DIMS[*]}"
echo "Target SAE grid: scales=${SAE_SCALING_FACTORS[*]} | k=${SAE_TOPKS[*]}"
echo "Target SAE training: epochs=${SAE_EPOCHS}, early_stopping=${SAE_EARLY_STOPPING}"
echo

select_best_elsa_ckpt_for_dim() {
    local dim="$1"
    "${PYTHON_BIN}" - "${RESULTS_DIR}" "${CHECKPOINTS_DIR}" "${dim}" <<'PY'
import glob
import json
import os
import sys

results_dir, checkpoints_dir, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])

# Prefer best scored ELSA result JSON for this dimension.
best_json = None
best_score = float("-inf")
for fp in glob.glob(os.path.join(results_dir, f"ELSA-{dim}-*.json")):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        score = d["results"]["val"]["@10"]["ndcg"]["mean"]
    except Exception:
        continue
    if score > best_score:
        best_score = score
        best_json = fp

if best_json is not None:
    print(os.path.basename(best_json).replace(".json", ".ckpt"))
    raise SystemExit(0)

# Fallback: newest checkpoint for this dimension.
matches = glob.glob(os.path.join(checkpoints_dir, f"ELSA-{dim}-*.ckpt"))
if not matches:
    raise SystemExit(1)
matches.sort(key=os.path.getmtime)
print(os.path.basename(matches[-1]))
PY
}

existing_result_for_cfg() {
    local pretrained_ckpt="$1"
    local sae_dim="$2"
    local k="$3"
    "${PYTHON_BIN}" - "${RESULTS_DIR}" "${DATASET}" "${pretrained_ckpt}" "${sae_dim}" "${k}" "${SAE_RECON_LOSS}" "${SAE_LR}" "${SAE_L1_COEF}" "${SAE_EPOCHS}" "${SAE_EARLY_STOPPING}" <<'PY'
import glob
import json
import math
import os
import sys

(
    results_dir,
    dataset,
    pretrained_ckpt,
    sae_dim,
    k,
    recon_loss,
    lr,
    l1,
    epochs,
    early_stopping,
) = sys.argv[1:]

sae_dim = int(sae_dim)
k = int(k)
lr = float(lr)
l1 = float(l1)
epochs = int(epochs)
early_stopping = int(early_stopping)

def feq(a, b, eps=1e-12):
    return abs(float(a) - float(b)) <= eps

for fp in glob.glob(os.path.join(results_dir, f"TopKSAE-{sae_dim}-*.json")):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = d.get("job_cfg", {})
    except Exception:
        continue

    if cfg.get("dataset") != dataset:
        continue
    if cfg.get("pretrained_model_checkpoint") != pretrained_ckpt:
        continue
    if int(cfg.get("embedding_dim", -1)) != sae_dim:
        continue
    if int(cfg.get("k", -1)) != k:
        continue
    if cfg.get("reconstruction_loss") != recon_loss:
        continue
    if int(cfg.get("epochs", -1)) != epochs:
        continue
    if int(cfg.get("early_stopping", -1)) != early_stopping:
        continue
    if not feq(cfg.get("lr", 0.0), lr):
        continue
    if not feq(cfg.get("l1_coef", 0.0), l1):
        continue

    print(fp)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

for input_dim in "${TARGET_INPUT_DIMS[@]}"; do
    echo "=== Input dim ${input_dim} ==="

    if ! pretrained_ckpt="$(select_best_elsa_ckpt_for_dim "${input_dim}")"; then
        echo "No ELSA checkpoint found for input_dim=${input_dim}. Skipping."
        echo
        continue
    fi
    echo "Using ELSA checkpoint: ${pretrained_ckpt}"

    for scale in "${SAE_SCALING_FACTORS[@]}"; do
        sae_dim=$((input_dim * scale))
        for k in "${SAE_TOPKS[@]}"; do
            if existing_path="$(existing_result_for_cfg "${pretrained_ckpt}" "${sae_dim}" "${k}")"; then
                echo "Skip existing: input_dim=${input_dim}, sae_dim=${sae_dim}, k=${k} -> ${existing_path}"
                continue
            fi

            echo "Train missing: input_dim=${input_dim}, sae_dim=${sae_dim}, k=${k}"
            SAVE_CKPT="${SAVE_CKPT}" "${PYTHON_BIN}" train_sae.py \
                --dataset "${DATASET}" \
                --pretrained_model_checkpoint "${pretrained_ckpt}" \
                --model_class TopKSAE \
                --embedding_dim "${sae_dim}" \
                --reconstruction_loss "${SAE_RECON_LOSS}" \
                --lr "${SAE_LR}" \
                --l1_coef "${SAE_L1_COEF}" \
                --k "${k}" \
                --epochs "${SAE_EPOCHS}" \
                --early_stopping "${SAE_EARLY_STOPPING}" \
                | tee "${LOGS_DIR}/backfill_sae_in${input_dim}_dim${sae_dim}_k${k}.log"
        done
    done
    echo
done

echo "=== BACKFILL MISSING MEASUREMENTS END ==="
echo "Logs: ${LOGS_DIR}"
echo "Results JSON: ${RESULTS_DIR}"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
