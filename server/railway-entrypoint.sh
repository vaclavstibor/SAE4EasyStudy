#!/bin/sh
set -eu

cd /app/server

# ---------------------------------------------------------------------------
# 1. Dataset: ml-32m-filtered (CSVs + poster images)
# ---------------------------------------------------------------------------
if [ ! -f "${ML_DATASET_READY_FILE:-static/datasets/ml-32m-filtered/ratings.csv}" ]; then
  python bootstrap_datasets.py --tag "${DATASET_RELEASE_TAG:-latest}"
fi

# Remove archive after extraction to save volume space
rm -f static/datasets/ml-32m-filtered/ml-32m-filtered.zip

# ---------------------------------------------------------------------------
# 2. SAE model artifacts (checkpoint, runtime features, labels, embeddings,
#    semantic merged labels)
# ---------------------------------------------------------------------------
DATA_DIR="plugins/sae_steering/data"

MODEL_OUTPUT_PATH="${SAE_MODEL_OUTPUT_PATH:-${DATA_DIR}/TopKSAE-1024.ckpt}"
RUNTIME_OUTPUT_PATH="${SAE_RUNTIME_OUTPUT_PATH:-${DATA_DIR}/item_sae_features_TopKSAE-1024.pt}"
LABEL_OUTPUT_PATH="${SAE_LABEL_OUTPUT_PATH:-${DATA_DIR}/llm_labels_TopKSAE-1024_llm.json}"
EMBEDDINGS_PATH="${SAE_EMBEDDINGS_PATH:-${DATA_DIR}/item_embeddings.pt}"
SEMANTIC_MERGED_PATH="${SAE_SEMANTIC_MERGED_PATH:-${DATA_DIR}/semantic_merged_TopKSAE-1024.json}"

MODEL_CKPT_PATH="${MODEL_OUTPUT_PATH}"
MODEL_PT_PATH="${MODEL_OUTPUT_PATH%.ckpt}.pt"

NEED_BOOTSTRAP=false
if [ ! -f "$MODEL_CKPT_PATH" ] && [ ! -f "$MODEL_PT_PATH" ]; then NEED_BOOTSTRAP=true; fi
if [ ! -f "$RUNTIME_OUTPUT_PATH" ]; then NEED_BOOTSTRAP=true; fi
if [ ! -f "$LABEL_OUTPUT_PATH" ]; then NEED_BOOTSTRAP=true; fi
if [ ! -f "$EMBEDDINGS_PATH" ]; then NEED_BOOTSTRAP=true; fi
if [ ! -f "$SEMANTIC_MERGED_PATH" ]; then NEED_BOOTSTRAP=true; fi

if [ "$NEED_BOOTSTRAP" = true ]; then
  python plugins/sae_steering/bootstrap_model.py \
    --tag "${SAE_MODEL_RELEASE_TAG:-latest}" \
    --model-asset-name "${SAE_MODEL_ASSET_NAME:-}" \
    --runtime-asset-name "${SAE_RUNTIME_ASSET_NAME:-item_sae_features_TopKSAE-1024.pt.xz}" \
    --label-asset-name "${SAE_LABEL_ASSET_NAME:-}" \
    --label-optional \
    --model-output "$MODEL_OUTPUT_PATH" \
    --runtime-output "$RUNTIME_OUTPUT_PATH" \
    --label-output "$LABEL_OUTPUT_PATH"
fi

echo "[startup] SAE data dir contents:"
ls -lah "$DATA_DIR" || true
if [ ! -f "$LABEL_OUTPUT_PATH" ]; then
  echo "[startup] LLM label cache missing: $LABEL_OUTPUT_PATH"
fi

# ---------------------------------------------------------------------------
# 3. Ensure the model is visible under plugins/sae_steering/models/
# ---------------------------------------------------------------------------
mkdir -p plugins/sae_steering/models
if [ -f "$MODEL_CKPT_PATH" ] && [ ! -f "plugins/sae_steering/models/TopKSAE-1024.ckpt" ]; then
  ln -s "$(realpath "$MODEL_CKPT_PATH")" "plugins/sae_steering/models/TopKSAE-1024.ckpt" || \
    cp "$MODEL_CKPT_PATH" "plugins/sae_steering/models/TopKSAE-1024.ckpt"
fi
if [ -f "$MODEL_PT_PATH" ] && [ ! -f "plugins/sae_steering/models/TopKSAE-1024.pt" ]; then
  ln -s "$(realpath "$MODEL_PT_PATH")" "plugins/sae_steering/models/TopKSAE-1024.pt" || \
    cp "$MODEL_PT_PATH" "plugins/sae_steering/models/TopKSAE-1024.pt"
fi

# ---------------------------------------------------------------------------
# 4. Start gunicorn
# ---------------------------------------------------------------------------
exec python -m gunicorn \
  -w "${GUNICORN_WORKERS:-1}" \
  --bind "0.0.0.0:${PORT:-5000}" \
  --timeout "${GUNICORN_TIMEOUT:-0}" \
  --preload \
  --log-level "${GUNICORN_LOG_LEVEL:-info}" \
  --access-logfile - \
  --error-logfile - \
  "app:create_app()"
