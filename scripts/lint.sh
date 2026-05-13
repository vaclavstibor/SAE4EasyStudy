#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

exec ./server/.venv39/bin/python -m ruff check \
  server/platform \
  server/plugins/fastcompare \
  server/plugins/empty_template \
  server/plugins/steering/recommendation/model_store.py \
  server/scripts/init_db.py \
  tests \
  "$@"
