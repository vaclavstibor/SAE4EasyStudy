#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

exec ./server/.venv39/bin/python -m gunicorn \
  -w "${GUNICORN_WORKERS:-1}" \
  --bind "0.0.0.0:${PORT:-5000}" \
  --timeout "${GUNICORN_TIMEOUT:-0}" \
  --preload \
  --log-level "${GUNICORN_LOG_LEVEL:-info}" \
  --access-logfile - \
  --error-logfile - \
  "server.platform.app:create_app()"
