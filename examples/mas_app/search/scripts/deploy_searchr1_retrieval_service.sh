#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/searchr1_download.py"
RETRIEVAL_SERVER_SCRIPT="${SCRIPT_DIR}/retrieval_server.py"
# Environment-variable configurable defaults:
#   SEARCH_MAS_SEARCHR1_LOCAL_DIR
#   SEARCH_MAS_SEARCHR1_PORT
#   SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL
LOCAL_DIR="${SEARCH_MAS_SEARCHR1_LOCAL_DIR:-${LOCAL_DIR:-/data1/lll/datasets/wiki-18}}"
CONDA_ENV="retriever"
PORT="${SEARCH_MAS_SEARCHR1_PORT:-${PORT:-8010}}"
RETRIEVER_MODEL="${SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL:-${RETRIEVER_MODEL:-/data1/lll/models/e5-base-v2}}"
LOG_FILE="retrieval_server.log"
SETUP_ENV="1"
FORCE_ENV_SETUP="0"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/deploy_searchr1_retrieval_service.sh [options]

Options:
  --local-dir PATH    Directory to store index/corpus files (env: SEARCH_MAS_SEARCHR1_LOCAL_DIR, default: ${LOCAL_DIR}).
  --conda-env NAME    Conda env name for retriever (default: retriever).
  --port PORT         Retrieval service port (env: SEARCH_MAS_SEARCHR1_PORT, default: ${PORT}).
  --retriever-model PATH
                     Retrieval encoder model path (env: SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL, default: ${RETRIEVER_MODEL}).
  --log-file PATH     Log file path (default: retrieval_server.log).
  --skip-env-setup    Skip conda env creation/dependency installation.
  --force-env-setup   Reinstall retriever dependencies even if env already exists.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --retriever-model)
      RETRIEVER_MODEL="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --skip-env-setup)
      SETUP_ENV="0"
      shift
      ;;
    --force-env-setup)
      FORCE_ENV_SETUP="1"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${DOWNLOAD_SCRIPT}" ]]; then
  echo "Missing local script: ${DOWNLOAD_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${RETRIEVAL_SERVER_SCRIPT}" ]]; then
  echo "Missing local script: ${RETRIEVAL_SERVER_SCRIPT}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install conda and create env '${CONDA_ENV}' first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

ensure_retriever_env() {
  local env_exists="0"
  if conda env list | awk 'NR>2 {print $1}' | grep -Fxq "${CONDA_ENV}"; then
    env_exists="1"
  fi

  local need_install="0"
  if [[ "${env_exists}" == "0" ]]; then
    echo "Creating conda env: ${CONDA_ENV}"
    conda create -n "${CONDA_ENV}" python=3.10 -y
    need_install="1"
  elif [[ "${FORCE_ENV_SETUP}" == "1" ]]; then
    need_install="1"
  fi

  conda activate "${CONDA_ENV}"

  if [[ "${need_install}" == "0" ]]; then
    echo "Conda env '${CONDA_ENV}' already exists. Skipping dependency installation."
    echo "Use --force-env-setup to reinstall retriever dependencies."
    return
  fi

  echo "Installing retriever dependencies in conda env '${CONDA_ENV}'"
  conda install numpy==1.26.4 -y
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install transformers datasets pyserini huggingface_hub
  conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
  pip install uvicorn fastapi
}

if [[ "${SETUP_ENV}" == "1" ]]; then
  echo "[1/4] Ensuring retriever conda environment: ${CONDA_ENV}"
  ensure_retriever_env
else
  if ! conda env list | awk 'NR>2 {print $1}' | grep -Fxq "${CONDA_ENV}"; then
    echo "Conda env '${CONDA_ENV}' does not exist. Remove --skip-env-setup or create it first." >&2
    exit 1
  fi
  echo "[1/4] Skipping env setup. Activating conda env: ${CONDA_ENV}"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "${LOCAL_DIR}"

echo "[2/4] Downloading index and corpus to ${LOCAL_DIR}"
# python "${DOWNLOAD_SCRIPT}" --local_dir "${LOCAL_DIR}"

echo "[3/4] Building e5_Flat.index and extracting wiki-18 corpus"
# shopt -s nullglob
# parts=( "${LOCAL_DIR}"/part_* )
# if [[ ${#parts[@]} -eq 0 ]]; then
#   echo "No part_* files found under ${LOCAL_DIR}" >&2
#   exit 1
# fi
# cat "${parts[@]}" > "${LOCAL_DIR}/e5_Flat.index"

# if [[ -f "${LOCAL_DIR}/wiki-18.jsonl.gz" ]]; then
#   gzip -df "${LOCAL_DIR}/wiki-18.jsonl.gz"
# fi

if [[ ! -f "${LOCAL_DIR}/wiki-18.jsonl" ]]; then
  echo "Corpus file missing: ${LOCAL_DIR}/wiki-18.jsonl" >&2
  exit 1
fi

echo "[4/4] Starting retrieval server on port ${PORT}. Logs: ${LOG_FILE}"
echo "      Local dir: ${LOCAL_DIR}"
echo "      Retriever model: ${RETRIEVER_MODEL}"
python "${RETRIEVAL_SERVER_SCRIPT}" \
  --index_path "${LOCAL_DIR}/e5_Flat.index" \
  --corpus_path "${LOCAL_DIR}/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --faiss_gpu \
  --port "${PORT}" \
  > "${LOG_FILE}"
