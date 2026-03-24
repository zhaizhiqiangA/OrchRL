#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/scripts/utils/export_repo_pythonpath.sh"

CONFIG_NAME="${CONFIG_NAME:-search_mas_tree_real_smoke}"
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8010/retrieve}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/search_mas_tree_real_smoke.log}"

exec python3 "$REPO_ROOT/examples/search_mas_tree/run.py" \
  --config-name "$CONFIG_NAME" \
  --retrieval-service-url "$RETRIEVAL_SERVICE_URL" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --log-path "$LOG_PATH" \
  "$@"
