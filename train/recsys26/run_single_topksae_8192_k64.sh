#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-ml-32m-filtered}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_CKPT="${SAVE_CKPT:-True}"

# Target single-run config
SAE_DIM="${SAE_DIM:-8192}"
SAE_K="${SAE_K:-64}"
SAE_EPOCHS="${SAE_EPOCHS:-120}"
SAE_EARLY_STOPPING="${SAE_EARLY_STOPPING:-20}"
SAE_LR="${SAE_LR:-3e-4}"
SAE_L1_COEF="${SAE_L1_COEF:-0.0003}"
SAE_RECON_LOSS="${SAE_RECON_LOSS:-Cosine}"

ALT_DATASET="${DATASET}"
if [[ "${DATASET}" == "ml-32-m-filtered" ]]; then
    ALT_DATASET="ml-32m-filtered"
elif [[ "${DATASET}" == "ml-32m-filtered" ]]; then
    ALT_DATASET="ml-32-m-filtered"
fi

EFFECTIVE_DATASET="${DATASET}"
if [[ ! -d "checkpoints/${EFFECTIVE_DATASET}" && -d "checkpoints/${ALT_DATASET}" ]]; then
    EFFECTIVE_DATASET="${ALT_DATASET}"
fi
if [[ ! -e "checkpoints/${EFFECTIVE_DATASET}/ELSA-"*.ckpt ]] && [[ -e "checkpoints/${ALT_DATASET}/ELSA-"*.ckpt ]]; then
    EFFECTIVE_DATASET="${ALT_DATASET}"
fi

RESULTS_DIR="results/${EFFECTIVE_DATASET}"
CHECKPOINTS_DIR="checkpoints/${EFFECTIVE_DATASET}"
LOGS_DIR="logs/${EFFECTIVE_DATASET}"

mkdir -p "${RESULTS_DIR}" "${CHECKPOINTS_DIR}" "${LOGS_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter '${PYTHON_BIN}' not found."
    exit 1
fi

echo "=== SINGLE SAE RUN START ==="
echo "Requested dataset: ${DATASET}"
echo "Effective dataset: ${EFFECTIVE_DATASET}"
echo "SAE dim: ${SAE_DIM}"
echo "SAE k: ${SAE_K}"
echo

# Optional manual override:
# export PRETRAINED_ELSA_CKPT=ELSA-2048-xxxxxx.ckpt
PRETRAINED_ELSA_CKPT="${PRETRAINED_ELSA_CKPT:-}"

if [[ -z "${PRETRAINED_ELSA_CKPT}" ]]; then
    echo "Selecting best ELSA checkpoint from ${RESULTS_DIR} (val nDCG@10)..."
    set +e
    PRETRAINED_ELSA_CKPT="$("${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY'
import glob
import json
import os
import sys

results_dir = sys.argv[1]
best_file = None
best_score = float("-inf")

for fp in glob.glob(os.path.join(results_dir, "ELSA-*.json")):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data["results"]["val"]["@10"]["ndcg"]["mean"]
    except Exception:
        continue
    if score > best_score:
        best_score = score
        best_file = fp

if best_file is None:
    raise SystemExit(1)

ckpt = os.path.basename(best_file).replace(".json", ".ckpt")
print(ckpt)
PY
)"
    select_status=$?
    set -e

    if [[ ${select_status} -ne 0 || -z "${PRETRAINED_ELSA_CKPT}" ]]; then
        echo "No valid ELSA result JSON found by metric parsing."
        echo "Trying ELSA JSON filename fallback in ${RESULTS_DIR}..."
        set +e
        PRETRAINED_ELSA_CKPT="$("${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY'
import glob
import os
import re
import sys

results_dir = sys.argv[1]
matches = glob.glob(os.path.join(results_dir, "ELSA-*.json"))
if not matches:
    raise SystemExit(1)

def key(fp: str):
    name = os.path.basename(fp)
    m = re.match(r"ELSA-(\d+)-", name)
    dim = int(m.group(1)) if m else -1
    return (dim, os.path.getmtime(fp))

matches.sort(key=key)
print(os.path.basename(matches[-1]).replace(".json", ".ckpt"))
PY
)"
        select_status=$?
        set -e
    fi

    if [[ ${select_status} -ne 0 || -z "${PRETRAINED_ELSA_CKPT}" ]]; then
        echo "No usable ELSA JSON found. Falling back to checkpoints in ${CHECKPOINTS_DIR}..."
        PRETRAINED_ELSA_CKPT="$("${PYTHON_BIN}" - "${CHECKPOINTS_DIR}" <<'PY'
import glob
import os
import re
import sys

checkpoints_dir = sys.argv[1]
matches = glob.glob(os.path.join(checkpoints_dir, "ELSA-*.ckpt"))
if not matches:
    raise SystemExit("No ELSA checkpoints found in checkpoints directory.")

def key(fp: str):
    name = os.path.basename(fp)
    m = re.match(r"ELSA-(\d+)-", name)
    dim = int(m.group(1)) if m else -1
    return (dim, os.path.getmtime(fp))

matches.sort(key=key)
print(os.path.basename(matches[-1]))
PY
)"
        echo "Fallback selected ELSA checkpoint: ${PRETRAINED_ELSA_CKPT}"
    fi
fi

if [[ ! -f "${CHECKPOINTS_DIR}/${PRETRAINED_ELSA_CKPT}" ]]; then
    echo "ELSA checkpoint not found: ${CHECKPOINTS_DIR}/${PRETRAINED_ELSA_CKPT}"
    echo "Train ELSA first or set PRETRAINED_ELSA_CKPT correctly (basename only)."
    exit 1
fi

echo "Using ELSA checkpoint: ${PRETRAINED_ELSA_CKPT}"
echo

SAVE_CKPT="${SAVE_CKPT}" "${PYTHON_BIN}" train_sae.py \
    --dataset "${EFFECTIVE_DATASET}" \
    --pretrained_model_checkpoint "${PRETRAINED_ELSA_CKPT}" \
    --model_class TopKSAE \
    --embedding_dim "${SAE_DIM}" \
    --reconstruction_loss "${SAE_RECON_LOSS}" \
    --lr "${SAE_LR}" \
    --l1_coef "${SAE_L1_COEF}" \
    --k "${SAE_K}" \
    --epochs "${SAE_EPOCHS}" \
    --early_stopping "${SAE_EARLY_STOPPING}" \
    | tee "${LOGS_DIR}/single_sae_8192_k64.log"

latest_result="$("${PYTHON_BIN}" - "${RESULTS_DIR}" "${SAE_DIM}" <<'PY'
import glob
import os
import sys

results_dir, sae_dim = sys.argv[1], sys.argv[2]
matches = glob.glob(os.path.join(results_dir, f"TopKSAE-{sae_dim}-*.json"))
if not matches:
    raise SystemExit("No TopKSAE result JSON found.")
matches.sort(key=os.path.getmtime)
print(matches[-1])
PY
)"

echo
echo "=== SINGLE SAE RUN END ==="
echo "Result JSON: ${latest_result}"
echo "Log file: ${LOGS_DIR}/single_sae_8192_k64.log"
echo "Notebook will include it automatically from ${RESULTS_DIR}."
