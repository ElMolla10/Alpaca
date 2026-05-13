#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# start DriftWatch postgres container if it's stopped
docker start pg-obs >/dev/null 2>&1 || true

# python env
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

# driftwatch db (env override if you already exported it)
export DRIFTWATCH_DATABASE_URL="${DRIFTWATCH_DATABASE_URL:-postgresql://postgres:password@127.0.0.1:5432/driftwatch}"

python app/main.py
