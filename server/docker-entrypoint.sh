#!/bin/sh
set -eu

# ---------------------------------------------------------------------------
# Volume persistence (Railway / Docker volume).
# When DATA_ROOT=/data is mounted as a persistent volume, we symlink the
# five state directories into it so they survive container redeploys.
# In plain Docker or local runs DATA_ROOT is empty and this block is a no-op.
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/data}"
if [ -d "${DATA_ROOT}" ] && [ "${DATA_ROOT}" != "/" ]; then
  for rel in instance cache plugins/steering/models plugins/steering/data; do
    src="${DATA_ROOT}/${rel}"
    dst="/app/server/${rel}"
    mkdir -p "${src}"
    if [ ! -L "${dst}" ]; then
      rm -rf "${dst}"
      ln -s "${src}" "${dst}"
    fi
  done
  mkdir -p "${DATA_ROOT}/datasets"
  if [ ! -L "/app/server/static/datasets" ]; then
    rm -rf "/app/server/static/datasets"
    ln -s "${DATA_ROOT}/datasets" "/app/server/static/datasets"
  fi
fi

mkdir -p /app/server/instance /app/server/cache /app/server/plugins/steering/models /app/server/plugins/steering/data

DATASET_DIR="/app/server/static/datasets/ml-32m-filtered"
MODEL_DIR="/app/server/plugins/steering/models"
DATA_DIR="/app/server/plugins/steering/data"

die_missing_asset() {
  echo "ERROR: missing required asset: $1" >&2
  echo "See README.md -> Manual local asset placement for the study deployment." >&2
  exit 1
}

if [ "${DATASET_BOOTSTRAP:-0}" = "1" ]; then
  if [ -f "${DATASET_DIR}/ratings.csv" ]; then
    echo "Dataset already present at ${DATASET_DIR}, skipping download."
  else
    echo "Bootstrapping dataset from GitHub Releases..."
    python -m server.plugins.steering.bootstrap_dataset
  fi
fi

if [ "${SAE_BOOTSTRAP_MODEL:-0}" = "1" ]; then
  if [ -f "${MODEL_DIR}/TopKSAE-1024.ckpt" ] || [ -f "${MODEL_DIR}/TopKSAE-1024.pt" ]; then
    echo "SAE model already present at ${MODEL_DIR}, skipping download."
  else
    echo "Bootstrapping SAE steering model from GitHub Releases..."
    python -m server.plugins.steering.bootstrap_model
  fi
fi

[ -f "${DATASET_DIR}/ratings.csv" ] || die_missing_asset "${DATASET_DIR}/ratings.csv"
[ -f "${DATASET_DIR}/movies.csv" ] || die_missing_asset "${DATASET_DIR}/movies.csv"
[ -f "${DATASET_DIR}/plots.csv" ] || die_missing_asset "${DATASET_DIR}/plots.csv"

if [ ! -f "${MODEL_DIR}/TopKSAE-1024.ckpt" ] && [ ! -f "${MODEL_DIR}/TopKSAE-1024.pt" ]; then
  die_missing_asset "${MODEL_DIR}/TopKSAE-1024.ckpt (or .pt)"
fi

[ -f "${DATA_DIR}/item_sae_features_TopKSAE-1024.pt" ] || die_missing_asset "${DATA_DIR}/item_sae_features_TopKSAE-1024.pt"
[ -f "${DATA_DIR}/item_embeddings.pt" ] || die_missing_asset "${DATA_DIR}/item_embeddings.pt"
[ -f "${DATA_DIR}/llm_labels_TopKSAE-1024_llm.json" ] || die_missing_asset "${DATA_DIR}/llm_labels_TopKSAE-1024_llm.json"
[ -f "${DATA_DIR}/semantic_merged_TopKSAE-1024.json" ] || die_missing_asset "${DATA_DIR}/semantic_merged_TopKSAE-1024.json"

cd /app

echo "Initializing database schema..."
python /app/server/scripts/init_db.py

exec python -m gunicorn \
  -w "${GUNICORN_WORKERS:-1}" \
  --bind "0.0.0.0:${PORT:-5000}" \
  --timeout "${GUNICORN_TIMEOUT:-0}" \
  --preload \
  --log-level "${GUNICORN_LOG_LEVEL:-info}" \
  --access-logfile - \
  --error-logfile - \
  "server.platform.app:create_app()"
