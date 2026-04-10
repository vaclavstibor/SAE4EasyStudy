#!/bin/sh
set -eu

mkdir -p /app/instance /app/cache /app/plugins/sae_steering/models

if [ "${SAE_BOOTSTRAP_MODEL:-1}" = "1" ]; then
  echo "Bootstrapping SAE steering model from GitHub Releases..."
  python plugins/sae_steering/bootstrap_model.py
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
