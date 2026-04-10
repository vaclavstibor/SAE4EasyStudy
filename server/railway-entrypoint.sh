#!/bin/sh
set -eu

cd /app/server

# Download dataset assets only on first run (or when missing)
if [ ! -f "${ML_DATASET_READY_FILE:-static/datasets/ml-latest/ratings.csv}" ]; then
  python bootstrap_datasets.py --tag "${DATASET_RELEASE_TAG:-latest}"
fi

# Remove archive after extraction to save volume space
rm -f static/datasets/ml-latest/ml-latest.zip

# Download SAE runtime features only on first run (or when missing)
if [ ! -f "${SAE_RUNTIME_OUTPUT_PATH:-plugins/sae_steering/data/item_sae_features_www_TopKSAE_8192.pt}" ]; then
  python plugins/sae_steering/bootstrap_model.py \
    --tag "${SAE_MODEL_RELEASE_TAG:-latest}" \
    --runtime-asset-name "${SAE_RUNTIME_ASSET_NAME:-item_sae_features_www_TopKSAE_8192.pt.xz}"
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
