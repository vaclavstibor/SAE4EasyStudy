#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

exec ./server/.venv39/bin/python server/scripts/init_db.py
