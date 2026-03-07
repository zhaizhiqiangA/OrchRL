#!/usr/bin/env bash

# Source this file before running Search MAS:
#   source scripts/env_mas_runtime.sh
#   python scripts/run_search_mas.py --config configs/search_mas_example.yaml --question "..."

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script instead: source scripts/env_mas_runtime.sh" >&2
  exit 1
fi

# Core LLM settings for Search MAS runtime.
export SEARCH_MAS_LLM_BASE_URL="${SEARCH_MAS_LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
export SEARCH_MAS_LLM_API_KEY="${SEARCH_MAS_LLM_API_KEY:-${OPENAI_API_KEY:-}}"
export SEARCH_MAS_LLM_MODEL="${SEARCH_MAS_LLM_MODEL:-${OPENAI_MODEL:-gpt-4.1-mini}}"

# Search backend URL consumed by search.retrieval_service_url.
export SEARCH_MAS_RETRIEVAL_SERVICE_URL="${SEARCH_MAS_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8010/retrieve}"

# Keep OpenAI-compatible aliases in sync for external tools/scripts.
export OPENAI_BASE_URL="${SEARCH_MAS_LLM_BASE_URL}"
if [[ -n "${SEARCH_MAS_LLM_API_KEY}" ]]; then
  export OPENAI_API_KEY="${SEARCH_MAS_LLM_API_KEY}"
fi
export OPENAI_MODEL="${SEARCH_MAS_LLM_MODEL}"
