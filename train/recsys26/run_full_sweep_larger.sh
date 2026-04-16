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

# Larger ELSA sweep than run_full_sweep.sh
# Default is only 4096; override from shell if needed, e.g.:
#   ELSA_DIMS="1024 2048 4096 8192" bash run_full_sweep_larger.sh
ELSA_DIMS_STR="${ELSA_DIMS:-4096}"
read -r -a ELSA_DIMS <<< "${ELSA_DIMS_STR}"
ELSA_EPOCHS="${ELSA_EPOCHS:-30}"
ELSA_EARLY_STOPPING="${ELSA_EARLY_STOPPING:-12}"
ELSA_LR="${ELSA_LR:-3e-4}"

# Keep SAE grid same as full sweep for comparability
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

echo "=== FULL SWEEP (LARGER) START ==="
echo "Dataset: ${DATASET}"
echo "Python: ${PYTHON_BIN}"
echo "Save checkpoints: ${SAVE_CKPT}"
echo "ELSA dims: ${ELSA_DIMS[*]}"
echo

get_latest_result_for_dim() {
    local dim="$1"
    ${PYTHON_BIN} - "${RESULTS_DIR}" "${dim}" <<'PY'
import glob
import os
import sys

results_dir, dim = sys.argv[1], sys.argv[2]
matches = glob.glob(os.path.join(results_dir, f"ELSA-{dim}-*.json"))
if not matches:
    raise SystemExit(1)
matches.sort(key=os.path.getmtime)
print(matches[-1])
PY
}

echo "=== Phase 1/2: ELSA larger sweep ==="
elsa_result_files=()
for dim in "${ELSA_DIMS[@]}"; do
    if compgen -G "${RESULTS_DIR}/ELSA-${dim}-*.json" > /dev/null; then
        echo "--- Skipping ELSA dim=${dim} (existing result found) ---"
        res_file="$(get_latest_result_for_dim "${dim}")"
        echo "Using existing ELSA result file: ${res_file}"
        elsa_result_files+=("${res_file}")
        continue
    fi
    echo "--- Training ELSA dim=${dim} ---"
    SAVE_CKPT="${SAVE_CKPT}" "${PYTHON_BIN}" train_elsa.py \
        --dataset "${DATASET}" \
        --embedding_dim "${dim}" \
        --lr "${ELSA_LR}" \
        --epochs "${ELSA_EPOCHS}" \
        --early_stopping "${ELSA_EARLY_STOPPING}" \
        --val_user_ratio 0.2 \
        --test_user_ratio 0.0 \
        --eval_topks 10,20 \
        | tee "${LOGS_DIR}/full_larger_elsa_${dim}.log"
    res_file="$(get_latest_result_for_dim "${dim}")"
    echo "ELSA dim=${dim} result file: ${res_file}"
    elsa_result_files+=("${res_file}")
done

best_elsa_result="$(${PYTHON_BIN} - "${elsa_result_files[@]}" <<'PY'
import json
import sys

best_file = None
best_score = float("-inf")
for fp in sys.argv[1:]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    score = data["results"]["val"]["@10"]["ndcg"]["mean"]
    print(f"Candidate ELSA: {fp} | val ndcg@10={score:.6f}", file=sys.stderr)
    if score > best_score:
        best_score = score
        best_file = fp
print(best_file)
PY
)"

best_elsa_checkpoint="$(basename "${best_elsa_result%.json}.ckpt")"
best_elsa_dim="$(echo "${best_elsa_checkpoint}" | cut -d'-' -f2)"
echo
echo "Best ELSA result: ${best_elsa_result}"
echo "Best ELSA checkpoint: ${best_elsa_checkpoint}"
echo

echo "=== Phase 2/2: TopKSAE full sweep over best ELSA ==="
sae_result_files=()
for scale in "${SAE_SCALING_FACTORS[@]}"; do
    sae_dim=$((best_elsa_dim * scale))
    for k in "${SAE_TOPKS[@]}"; do
        echo "--- Training TopKSAE sae_dim=${sae_dim} k=${k} ---"
        SAVE_CKPT="${SAVE_CKPT}" "${PYTHON_BIN}" train_sae.py \
            --dataset "${DATASET}" \
            --pretrained_model_checkpoint "${best_elsa_checkpoint}" \
            --model_class TopKSAE \
            --embedding_dim "${sae_dim}" \
            --reconstruction_loss "${SAE_RECON_LOSS}" \
            --lr "${SAE_LR}" \
            --l1_coef "${SAE_L1_COEF}" \
            --k "${k}" \
            --epochs "${SAE_EPOCHS}" \
            --early_stopping "${SAE_EARLY_STOPPING}" \
            | tee "${LOGS_DIR}/full_larger_sae_s${scale}_k${k}.log"

        sae_result="$(${PYTHON_BIN} - "${RESULTS_DIR}" "${sae_dim}" <<'PY'
import glob
import os
import sys

results_dir, sae_dim = sys.argv[1], sys.argv[2]
matches = glob.glob(os.path.join(results_dir, f"TopKSAE-{sae_dim}-*.json"))
if not matches:
    raise SystemExit(1)
matches.sort(key=os.path.getmtime)
print(matches[-1])
PY
)"
        echo "TopKSAE result file: ${sae_result}"
        sae_result_files+=("${sae_result}")
    done
done

best_sae_result="$(${PYTHON_BIN} - "${sae_result_files[@]}" <<'PY'
import json
import sys

best_file = None
best_score = float("-inf")
for fp in sys.argv[1:]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    score = data["results"]["val"]["ndcg degradation"]["mean"]
    print(f"Candidate SAE: {fp} | val ndcg degradation={score:.6f}", file=sys.stderr)
    if score > best_score:
        best_score = score
        best_file = fp
print(best_file)
PY
)"

echo
echo "=== FULL SWEEP (LARGER) END ==="
echo "Best ELSA checkpoint: ${best_elsa_checkpoint}"
echo "Best SAE result: ${best_sae_result}"
echo "Logs: ${LOGS_DIR}"
echo "Results JSON: ${RESULTS_DIR}"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
