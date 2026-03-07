#!/usr/bin/env bash

# Source this file before deploying the SearchR1 retriever service:
#   source scripts/env_retriever_service.sh
#   bash scripts/deploy_searchr1_retrieval_service.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script instead: source scripts/env_retriever_service.sh" >&2
  exit 1
fi

# Variables consumed by scripts/deploy_searchr1_retrieval_service.sh
export SEARCH_MAS_SEARCHR1_LOCAL_DIR="${SEARCH_MAS_SEARCHR1_LOCAL_DIR:-/data1/lll/datasets/wiki-18}"
export SEARCH_MAS_SEARCHR1_PORT="${SEARCH_MAS_SEARCHR1_PORT:-8010}"
export SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL="${SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL:-/data1/lll/models/e5-base-v2}"

# Keep MAS runtime retrieval URL aligned with retriever service port.
export SEARCH_MAS_RETRIEVAL_SERVICE_URL="${SEARCH_MAS_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${SEARCH_MAS_SEARCHR1_PORT}/retrieve}"
