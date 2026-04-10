#!/bin/sh
set -eu

cd /app/server

# Download dataset assets only on first run (or when missing)
if [ ! -f "${ML_DATASET_READY_FILE:-static/datasets/ml-latest/ratings.csv}" ]; then
  python bootstrap_datasets.py --tag "${DATASET_RELEASE_TAG:-latest}"
fi

# Remove archive after extraction to save volume space
rm -f static/datasets/ml-latest/ml-latest.zip

# Download SAE model checkpoint + runtime features only on first run (or when missing)
RUNTIME_OUTPUT_PATH="${SAE_RUNTIME_OUTPUT_PATH:-plugins/sae_steering/data/item_sae_features_www_TopKSAE_8192.pt}"
MODEL_OUTPUT_PATH="${SAE_MODEL_OUTPUT_PATH:-plugins/sae_steering/data/www_TopKSAE_8192.ckpt}"
LABEL_OUTPUT_PATH="${SAE_LABEL_OUTPUT_PATH:-plugins/sae_steering/data/llm_labels_www_TopKSAE_8192_llm.json}"
MODEL_CKPT_PATH="${MODEL_OUTPUT_PATH}"
MODEL_PT_PATH="${MODEL_OUTPUT_PATH%.ckpt}.pt"

MODEL_OUTPUT_ARG=""
RUNTIME_OUTPUT_ARG=""
if [ -n "$MODEL_OUTPUT_PATH" ]; then
  MODEL_OUTPUT_ARG="--model-output ${MODEL_OUTPUT_PATH}"
fi
if [ -n "$RUNTIME_OUTPUT_PATH" ]; then
  RUNTIME_OUTPUT_ARG="--runtime-output ${RUNTIME_OUTPUT_PATH}"
fi
if [ -n "$LABEL_OUTPUT_PATH" ]; then
  LABEL_OUTPUT_ARG="--label-output ${LABEL_OUTPUT_PATH}"
fi

if [ ! -f "$MODEL_CKPT_PATH" ] && [ ! -f "$MODEL_PT_PATH" ] || [ ! -f "$RUNTIME_OUTPUT_PATH" ] || [ ! -f "$LABEL_OUTPUT_PATH" ]; then
  python plugins/sae_steering/bootstrap_model.py \
    --tag "${SAE_MODEL_RELEASE_TAG:-latest}" \
    --model-asset-name "${SAE_MODEL_ASSET_NAME:-}" \
    --runtime-asset-name "${SAE_RUNTIME_ASSET_NAME:-item_sae_features_www_TopKSAE_8192.pt.xz}" \
    --label-asset-name "${SAE_LABEL_ASSET_NAME:-}" \
    --label-optional \
    $MODEL_OUTPUT_ARG \
    $RUNTIME_OUTPUT_ARG \
    $LABEL_OUTPUT_ARG
fi

echo "[startup] SAE data dir contents:"
ls -lah plugins/sae_steering/data || true
if [ ! -f "$LABEL_OUTPUT_PATH" ]; then
  echo "[startup] LLM label cache missing: $LABEL_OUTPUT_PATH"
fi

# Ensure the model is visible under plugins/sae_steering/models (expected by loader)
mkdir -p plugins/sae_steering/models
if [ -f "$MODEL_CKPT_PATH" ] && [ ! -f "plugins/sae_steering/models/www_TopKSAE_8192.ckpt" ]; then
  ln -s "$(realpath "$MODEL_CKPT_PATH")" "plugins/sae_steering/models/www_TopKSAE_8192.ckpt" || \
    cp "$MODEL_CKPT_PATH" "plugins/sae_steering/models/www_TopKSAE_8192.ckpt"
fi
if [ -f "$MODEL_PT_PATH" ] && [ ! -f "plugins/sae_steering/models/www_TopKSAE_8192.pt" ]; then
  ln -s "$(realpath "$MODEL_PT_PATH")" "plugins/sae_steering/models/www_TopKSAE_8192.pt" || \
    cp "$MODEL_PT_PATH" "plugins/sae_steering/models/www_TopKSAE_8192.pt"
fi

exec python -m gunicorn \
  -w "${GUNICORN_WORKERS:-1}" \
  --bind "0.0.0.0:${PORT:-5000}" \
  --timeout "${GUNICORN_TIMEOUT:-0}" \
  --preload \
  --log-level "${GUNICORN_LOG_LEVEL:-info}" \
  --access-logfile - \
  --error-logfile - \
  "app:create_app()"
