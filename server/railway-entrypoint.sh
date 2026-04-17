#!/bin/sh
set -eu

# ---------------------------------------------------------------------------
# 0. Persistent storage bootstrap
# ---------------------------------------------------------------------------
# Railway allows only ONE volume mount per service, so everything that needs
# to survive a redeploy lives under a single mount point (default: /data)
# with well-known subdirs symlinked into the canonical application paths.
# The script is idempotent: first boot creates the subdirs and migrates any
# pre-existing content from the image layer into the volume; subsequent
# boots just re-establish the symlinks.
# ---------------------------------------------------------------------------
PERSIST_ROOT="${PERSIST_ROOT:-/data}"

link_persistent_dir() {
  # $1 = subdir under $PERSIST_ROOT, $2 = canonical path inside the image
  src="${PERSIST_ROOT}/$1"
  dst="$2"
  mkdir -p "$src"
  if [ -L "$dst" ]; then
    # already linked, nothing to do
    return 0
  fi
  if [ -d "$dst" ]; then
    # migrate pre-existing contents from the image into the volume on first boot
    if [ -n "$(ls -A "$dst" 2>/dev/null || true)" ]; then
      cp -a "$dst/." "$src/" 2>/dev/null || true
    fi
    rm -rf "$dst"
  fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
}

if [ -d "$PERSIST_ROOT" ] && [ -w "$PERSIST_ROOT" ]; then
  echo "[startup] wiring persistent storage at $PERSIST_ROOT"
  link_persistent_dir "instance"   "/app/server/instance"
  link_persistent_dir "cache"      "/app/server/cache"
  link_persistent_dir "sae_data"   "/app/server/plugins/sae_steering/data"
  link_persistent_dir "sae_models" "/app/server/plugins/sae_steering/models"
  link_persistent_dir "backups"    "/app/backups"
else
  echo "[startup] no persistent mount at $PERSIST_ROOT — using ephemeral container storage"
  mkdir -p /app/backups /app/server/instance /app/server/cache \
           /app/server/plugins/sae_steering/data \
           /app/server/plugins/sae_steering/models
fi

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
# 4. Initialise the DB schema BEFORE the app starts taking traffic.
#    * Fresh Postgres / SQLite: creates tables via db.create_all() and
#      stamps Alembic head (legacy migrations carry SQLite-only syntax
#      that Postgres rejects, so we skip them on empty databases).
#    * Existing schema: defers to `flask db upgrade` for pending
#      migrations.  Override with SKIP_DB_UPGRADE=1 in an emergency.
# ---------------------------------------------------------------------------
if [ "${SKIP_DB_UPGRADE:-0}" != "1" ]; then
  echo "[startup] initialising database schema"
  python scripts/init_db.py || \
    echo "[startup] init_db.py failed (continuing — review logs)"
fi

# ---------------------------------------------------------------------------
# 5. Start gunicorn
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
